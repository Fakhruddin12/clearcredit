import numpy as np
import pandas as pd

from clearcredit.features import build_features, read_features


def test_sql_builds_applicant_level_history_without_future_rows(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    pd.DataFrame({
        "SK_ID_CURR": [1, 2], "TARGET": [0, 1], "AMT_INCOME_TOTAL": [50000., 30000.],
        "DAYS_BIRTH": [-15000, -18000], "CODE_GENDER": ["F", "M"],
    }).to_csv(raw / "application_train.csv", index=False)
    pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2], "DAYS_CREDIT": [-100, 20, -200],
        "AMT_CREDIT_SUM": [1000., 9999., 2000.], "CREDIT_DAY_OVERDUE": [0, 99, 1],
        "AMT_CREDIT_SUM_DEBT": [500., 9999., 800.],
    }).to_csv(raw / "bureau.csv", index=False)
    pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2], "DAYS_DECISION": [-50, 5, -20],
        "AMT_APPLICATION": [100., 999., 200.], "AMT_CREDIT": [90., 999., 180.],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Refused"],
    }).to_csv(raw / "previous_application.csv", index=False)
    cfg = {"data": {"application_train": "application_train.csv", "bureau": "bureau.csv", "previous_application": "previous_application.csv"}}
    path = build_features(raw, processed, cfg)
    result = read_features(path).set_index("SK_ID_CURR")
    assert result.loc[1, "bureau_n_prior_accounts"] == 1
    assert result.loc[1, "bureau_total_credit_sum"] == 1000
    assert result.loc[1, "prior_n_applications"] == 1
    assert result.loc[1, "prior_refusals"] == 0
