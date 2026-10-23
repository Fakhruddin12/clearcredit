import numpy as np
import pandas as pd
import pytest

from clearcredit.data import validate_application
from clearcredit.evaluation import binary_metrics, choose_threshold, expected_cost, subgroup_audit
from clearcredit.models import split_features


def test_validation_rejects_duplicate_applicants():
    df = pd.DataFrame({"SK_ID_CURR": [1, 1], "TARGET": [0, 1]})
    with pytest.raises(ValueError, match="unique"):
        validate_application(df)


def test_metrics_and_decision_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([.1, .2, .7, .9])
    metrics = binary_metrics(y, p, .5)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["tp"] == 2 and metrics["tn"] == 2
    threshold = choose_threshold(y, p, false_approval_cost=5, false_decline_cost=1)
    assert 0 < threshold < 1
    assert expected_cost(y, p, 5, 1, threshold) == 0


def test_sensitive_audit_fields_are_not_features():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0], "CODE_GENDER": ["F", "M", "F"],
        "DAYS_BIRTH": [-12000, -16000, -18000], "AMT_INCOME_TOTAL": [1., 2., 3.],
    })
    X, y = split_features(df)
    assert "CODE_GENDER" not in X
    assert "DAYS_BIRTH" not in X
    assert "AMT_INCOME_TOTAL" in X
    assert y.tolist() == [0, 1, 0]


def test_subgroup_audit_skips_tiny_cells():
    audit = subgroup_audit([0, 1], [.1, .8], ["a", "b"], .5)
    assert audit == {}
