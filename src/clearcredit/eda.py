from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def profile(frame: pd.DataFrame) -> dict:
    target = frame["TARGET"].astype(int)
    missing = frame.isna().mean().sort_values(ascending=False)
    numeric = frame.select_dtypes(include=[np.number]).drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    by_target = numeric.assign(_target=target).groupby("_target").median(numeric_only=True)
    top_shift = (by_target.loc[1] - by_target.loc[0]).abs().sort_values(ascending=False).head(20)
    categorical = frame.select_dtypes(include=["object", "string"])
    cat_summary = {}
    for col in categorical.columns[:25]:
        counts = frame[col].fillna("<MISSING>").value_counts().head(10)
        rates = frame.assign(_v=frame[col].fillna("<MISSING>"), _t=target).groupby("_v", dropna=False)['_t'].agg(["size", "mean"])
        cat_summary[col] = [{"category": str(v), "count": int(counts[v]), "default_rate": float(rates.loc[v, "mean"])} for v in counts.index]
    return {
        "row_count": int(len(frame)), "column_count": int(frame.shape[1]),
        "target_prevalence": float(target.mean()),
        "missingness_top": [{"feature": str(k), "fraction": float(v)} for k, v in missing.head(30).items()],
        "numeric_median_shift_top": [{"feature": str(k), "absolute_median_difference": float(v)} for k, v in top_shift.items()],
        "categorical_default_rates": cat_summary,
        "notes": ["Descriptive associations are not causal effects.", "Missingness may encode process and access differences; inspect before imputation."],
    }


def save_profile(frame: pd.DataFrame, output_path: str | Path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = profile(frame)
    path.write_text(json.dumps(data, indent=2))
    return data
