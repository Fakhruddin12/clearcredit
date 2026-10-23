from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


REQUIRED_APP_COLUMNS = {"SK_ID_CURR", "TARGET"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_application(frame: pd.DataFrame) -> dict:
    missing = REQUIRED_APP_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required application columns: {sorted(missing)}")
    if frame["SK_ID_CURR"].duplicated().any():
        raise ValueError("SK_ID_CURR must be unique in application_train")
    target = set(frame["TARGET"].dropna().unique())
    if target != {0, 1}:
        raise ValueError(f"Unexpected target values: {target}")
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "target_rate": float(frame["TARGET"].mean()),
        "duplicate_applicants": int(frame["SK_ID_CURR"].duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
    }


def load_features(path: str | Path) -> pd.DataFrame:
    from .features import read_features
    frame = read_features(path)
    validate_application(frame[["SK_ID_CURR", "TARGET"]])
    return frame
