import joblib
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from clearcredit.api import create_app


def test_scoring_api_uses_saved_model_bundle(tmp_path):
    model = DummyClassifier(strategy="prior").fit([[0.0], [1.0]], [0, 1])
    calibrator = DummyClassifier(strategy="prior").fit([[0.0], [1.0]], [0, 1])
    joblib.dump({"model": model, "calibrator": calibrator, "features": ["income_ratio"], "threshold": .4},
                tmp_path / "model.joblib")
    client = TestClient(create_app(tmp_path / "model.joblib"))
    response = client.post("/score", json={"features": {"income_ratio": 0.5}})
    assert response.status_code == 200
    assert response.json()["risk_score"] == .5
    assert response.json()["review_flag_at_research_threshold"] is True
