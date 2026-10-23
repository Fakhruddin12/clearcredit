from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def binary_metrics(y_true, probability, threshold=.5):
    from sklearn.metrics import confusion_matrix, precision_score, recall_score
    y, p = np.asarray(y_true), np.asarray(probability)
    labels = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, labels, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def expected_cost(y_true, probability, false_approval_cost=5.0, false_decline_cost=1.0, threshold=.5):
    y, p = np.asarray(y_true), np.asarray(probability)
    predicted_default = p >= threshold
    false_approval = int(np.sum((y == 1) & ~predicted_default))
    false_decline = int(np.sum((y == 0) & predicted_default))
    return float(false_approval_cost * false_approval + false_decline_cost * false_decline)


def choose_threshold(y, p, false_approval_cost=5.0, false_decline_cost=1.0):
    candidates = np.unique(np.quantile(p, np.linspace(.01, .99, 99)))
    costs = [(expected_cost(y, p, false_approval_cost, false_decline_cost, t), t) for t in candidates]
    return float(min(costs)[1])


def subgroup_audit(y, p, groups, threshold):
    """Descriptive audit only; do not interpret group differences as causality."""
    y, p, g = np.asarray(y), np.asarray(p), np.asarray(groups)
    out = {}
    for value in sorted(set(g.astype(str))):
        mask = g.astype(str) == value
        if mask.sum() < 50:
            continue
        defaults = y[mask] == 1
        decision = p[mask] >= threshold
        out[value] = {
            "n": int(mask.sum()),
            "observed_default_rate": float(y[mask].mean()),
            "mean_predicted_risk": float(p[mask].mean()),
            "false_negative_rate": float(np.mean(~decision[defaults])) if defaults.any() else None,
            "false_positive_rate": float(np.mean(decision[~defaults])) if (~defaults).any() else None,
            "brier": float(np.mean((p[mask] - y[mask]) ** 2)),
        }
    return out


def calibration_summary(y_true, probability, n_bins=10):
    from sklearn.calibration import calibration_curve
    y, p = np.asarray(y_true), np.asarray(probability)
    observed, predicted = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    eps = 1e-6
    logits = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    # Calibration intercept/slope from logistic recalibration; descriptive on held-out data.
    from sklearn.linear_model import LogisticRegression
    recal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logits.reshape(-1, 1), y)
    return {
        "calibration_intercept": float(recal.intercept_[0]),
        "calibration_slope": float(recal.coef_[0, 0]),
        "reliability_bins": [{"mean_predicted": float(a), "observed_rate": float(b)} for a, b in zip(predicted, observed)],
    }
