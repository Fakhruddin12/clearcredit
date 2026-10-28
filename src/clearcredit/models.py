from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def split_features(df, target="TARGET", id_column="SK_ID_CURR"):
    y = df[target].astype(int)
    # Both supported schemas must exclude identifiers and audit-only fields.
    # The saved v1 benchmark predates this UCI schema correction.
    audit_only = ["CODE_GENDER", "DAYS_BIRTH", "SEX", "AGE", "EDUCATION", "MARRIAGE"]
    X = df.drop(columns=[target, id_column, "ID", *audit_only], errors="ignore").copy()
    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].astype("object")
    return X, y


def logistic_pipeline(X):
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.columns.difference(numeric).tolist()
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler(with_mean=False))])
    categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))])
    prep = ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)], sparse_threshold=0.3)
    return Pipeline([("preprocess", prep), ("model", LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", solver="liblinear"))])


def make_models(X, seed=42, n_jobs=-1):
    models = {"logistic_regression": logistic_pipeline(X)}
    try:
        from lightgbm import LGBMClassifier
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical = X.columns.difference(numeric).tolist()
        prep = ColumnTransformer([
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical),
        ], sparse_threshold=0)
        models["lightgbm"] = Pipeline([("preprocess", prep), ("model", LGBMClassifier(
            n_estimators=700, learning_rate=0.03, num_leaves=24,
            max_depth=-1, min_child_samples=80, colsample_bytree=.85,
            subsample=.85, reg_lambda=5.0, class_weight="balanced",
            random_state=seed, n_jobs=n_jobs, verbosity=-1,
        ))])
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        models["catboost"] = Pipeline([
            ("preprocess", ColumnTransformer([
                ("num", SimpleImputer(strategy="median", add_indicator=True), X.select_dtypes(include=[np.number]).columns.tolist()),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), X.columns.difference(X.select_dtypes(include=[np.number]).columns).tolist()),
            ], sparse_threshold=0)),
            ("model", CatBoostClassifier(iterations=700, learning_rate=.03, depth=6,
              loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
              random_seed=seed, verbose=False, allow_writing_files=False)),
        ])
    except ImportError:
        pass
    if len(models) == 1:
        raise RuntimeError("Install LightGBM to run the multi-model comparison")
    return models


def feature_importance(model, X, y, sample_size=10000, seed=42):
    from sklearn.inspection import permutation_importance
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample_size, len(X)), replace=False)
    result = permutation_importance(model, X.iloc[idx], y.iloc[idx], scoring="average_precision", n_repeats=3, random_state=seed, n_jobs=1)
    order = np.argsort(result.importances_mean)[::-1][:30]
    return [{"feature": str(X.columns[i]), "mean_ap_drop": float(result.importances_mean[i]), "sd": float(result.importances_std[i])} for i in order]

