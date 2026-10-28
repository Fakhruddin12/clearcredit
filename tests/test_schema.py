import pandas as pd
from clearcredit.models import split_features


def test_uci_schema_keeps_demographics_and_identifiers_out_of_predictors():
    frame = pd.DataFrame({"SK_ID_CURR": [1, 2], "ID": [11, 12], "TARGET": [0, 1],
        "SEX": [1, 2], "AGE": [25, 60], "EDUCATION": [1, 2], "MARRIAGE": [1, 2],
        "LIMIT_BAL": [10000, 20000], "PAY_0": [0, 2]})
    X, y = split_features(frame)
    assert set(X) == {"LIMIT_BAL", "PAY_0"}
    assert y.tolist() == [0, 1]


def test_home_credit_schema_keeps_audit_fields_out_of_predictors():
    frame = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1],
        "CODE_GENDER": ["F", "M"], "DAYS_BIRTH": [-10000, -20000],
        "AMT_INCOME_TOTAL": [10000, 20000]})
    X, _ = split_features(frame)
    assert list(X) == ["AMT_INCOME_TOTAL"]
