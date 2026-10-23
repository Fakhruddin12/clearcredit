from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression

from .data import file_sha256, validate_application
from .evaluation import binary_metrics, calibration_summary, choose_threshold, expected_cost, subgroup_audit
from .features import build_features, read_features
from .models import feature_importance, make_models, split_features
from .eda import save_profile


def read_config(path):
    return yaml.safe_load(Path(path).read_text())


def prepare(cfg):
    output = build_features(cfg["data"]["raw_dir"], cfg["data"]["processed_dir"], cfg)
    df = read_features(output)
    checks = validate_application(df[["SK_ID_CURR", "TARGET"]])
    checks.update({"feature_file": str(output), "rows_after_join": int(len(df)), "columns_after_join": int(len(df.columns))})
    Path(cfg["paths"]["reports"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["reports"]) / "data_quality.json").write_text(json.dumps(checks, indent=2))
    save_profile(df, Path(cfg["paths"]["reports"]) / "eda_profile.json")
    return checks


def run_train(cfg):
    raw, processed = Path(cfg["data"]["raw_dir"]), Path(cfg["data"]["processed_dir"])
    app_csv = raw / cfg["data"].get("application_train", "application_train.csv")
    feature_path = processed / "application_features.parquet"
    if not feature_path.exists():
        prepare(cfg)
    frame = read_features(feature_path)
    checks = validate_application(frame[["SK_ID_CURR", "TARGET"]])
    X, y = split_features(frame)
    ids = np.arange(len(frame))
    dev_ids, test_ids = train_test_split(ids, test_size=float(cfg["validation"]["holdout_fraction"]), stratify=y, random_state=int(cfg["validation"]["seed"]))
    X_dev, y_dev = X.iloc[dev_ids].reset_index(drop=True), y.iloc[dev_ids].reset_index(drop=True)
    X_test, y_test = X.iloc[test_ids].reset_index(drop=True), y.iloc[test_ids].reset_index(drop=True)
    folds = StratifiedKFold(n_splits=int(cfg["validation"]["folds"]), shuffle=True, random_state=int(cfg["validation"]["seed"]))
    models = make_models(X_dev, int(cfg["validation"]["seed"]), int(cfg["model"]["n_jobs"]))
    scores, oof = {}, {}
    for name, prototype in models.items():
        prob = np.full(len(X_dev), np.nan)
        fold_scores = []
        for fold, (tr, va) in enumerate(folds.split(X_dev, y_dev), start=1):
            # Construct a fresh estimator per fold so no fitted state crosses folds.
            fold_model = make_models(X_dev, int(cfg["validation"]["seed"]), int(cfg["model"]["n_jobs"]))[name]
            fold_model.fit(X_dev.iloc[tr], y_dev.iloc[tr])
            fold_p = fold_model.predict_proba(X_dev.iloc[va])[:, 1]
            prob[va] = fold_p
            fold_scores.append(binary_metrics(y_dev.iloc[va], fold_p))
        oof[name] = prob
        scores[name] = {
            "fold_metrics": fold_scores,
            "mean_roc_auc": float(np.mean([m["roc_auc"] for m in fold_scores])),
            "sd_roc_auc": float(np.std([m["roc_auc"] for m in fold_scores], ddof=1)),
            "mean_pr_auc": float(np.mean([m["pr_auc"] for m in fold_scores])),
            "sd_pr_auc": float(np.std([m["pr_auc"] for m in fold_scores], ddof=1)),
            "mean_brier": float(np.mean([m["brier"] for m in fold_scores])),
        }
    # Select by CV PR-AUC (the outcome is imbalanced), with Brier used to flag poor risk calibration.
    selected = max(scores, key=lambda name: scores[name]["mean_pr_auc"])
    eps = 1e-6
    oof_raw = np.clip(oof[selected], eps, 1 - eps)
    oof_logits = np.log(oof_raw / (1 - oof_raw)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(oof_logits, y_dev)
    oof_calibrated = calibrator.predict_proba(oof_logits)[:, 1]
    threshold = choose_threshold(y_dev, oof_calibrated, float(cfg["decision"]["false_approval_cost"]), float(cfg["decision"]["false_decline_cost"]))
    final_model = make_models(X_dev, int(cfg["validation"]["seed"]), int(cfg["model"]["n_jobs"]))[selected]
    final_model.fit(X_dev, y_dev)
    test_raw = np.clip(final_model.predict_proba(X_test)[:, 1], eps, 1 - eps)
    test_logits = np.log(test_raw / (1 - test_raw)).reshape(-1, 1)
    test_p = calibrator.predict_proba(test_logits)[:, 1]
    # Use gender only for post-hoc auditing, never as a decision feature in this prototype.
    group = frame.iloc[test_ids]["CODE_GENDER"].fillna("Unknown").astype(str).to_numpy() if "CODE_GENDER" in frame else np.repeat("All", len(test_ids))
    test_metrics = binary_metrics(y_test, test_p, threshold)
    test_metrics["calibration"] = calibration_summary(y_test, test_p)
    test_metrics["scenario_cost"] = expected_cost(y_test, test_p, float(cfg["decision"]["false_approval_cost"]), float(cfg["decision"]["false_decline_cost"]), threshold)
    audit = subgroup_audit(y_test, test_p, group, threshold)
    if "DAYS_BIRTH" in frame:
        age = (-frame.iloc[test_ids]["DAYS_BIRTH"].astype(float) / 365.25).clip(lower=0)
        age_band = pd.cut(age, bins=[0, 25, 35, 45, 55, 120], labels=["18-24", "25-34", "35-44", "45-54", "55+"]).astype("object").fillna("Unknown")
        audit["age_band"] = subgroup_audit(y_test, test_p, age_band.to_numpy(), threshold)
    report = {
        "project": "ClearCredit", "rows": checks["rows"], "feature_count": int(X.shape[1]),
        "target_rate": checks["target_rate"], "split": {"development": len(dev_ids), "holdout": len(test_ids), "strategy": "stratified random; source has no clean prospective timestamp"},
        "cv": {"folds": int(cfg["validation"]["folds"]), "models": scores},
        "selected_model": selected, "threshold_selected_on_development_oof": threshold,
        "scenario_cost_assumptions": {"false_approval_cost": cfg["decision"]["false_approval_cost"], "false_decline_cost": cfg["decision"]["false_decline_cost"]},
        "holdout_metrics": test_metrics, "descriptive_subgroup_audit": audit,
        "data_sha256": file_sha256(app_csv) if app_csv.exists() else json.loads((processed / "source_manifest.json").read_text())["sha256"],
    }
    out = Path(cfg["paths"]["reports"]); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    (out / "model_card.json").write_text(json.dumps({
        "intended_use": "Educational risk-ranking and model-validation demonstration only",
        "prohibited_use": "Real lending eligibility, pricing, or adverse action decisions",
        "label": "Competition TARGET; not a universal default definition",
        "validation_limit": "Random stratified holdout, not prospective or out-of-time validation",
        "selected_model": selected,
        "known_risks": ["historical access bias", "geographic transportability", "missing data patterns", "calibration drift", "unverified legal compliance"],
    }, indent=2))
    artifact = Path(cfg["paths"]["artifacts"]); artifact.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "calibrator": calibrator, "features": list(X.columns), "threshold": threshold}, artifact / "model.joblib")
    importance = feature_importance(final_model, X_test, y_test)
    (out / "permutation_importance.json").write_text(json.dumps(importance, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(prog="clearcredit")
    parser.add_argument("command", choices=["prepare", "train", "report"])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.command == "prepare":
        print(json.dumps(prepare(cfg), indent=2))
    elif args.command == "report":
        metrics_path = Path(cfg["paths"]["reports"]) / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError("Run the training command before requesting its report")
        print(metrics_path.read_text())
    else:
        print(json.dumps(run_train(cfg), indent=2))


if __name__ == "__main__":
    main()
