"""Optional local scoring API; only intended for demonstration with synthetic cases."""
from pathlib import Path

import joblib


def create_app(artifact_path="artifacts/model.joblib"):
    from fastapi import FastAPI, HTTPException
    app = FastAPI(title="ClearCredit Research API", version="0.1.0")
    artifact_file = Path(artifact_path)
    bundle = joblib.load(artifact_file) if artifact_file.exists() else None

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": bundle is not None,
                "intended_use": "research demonstration only; not a lending decision service"}

    @app.post("/score")
    def score(payload: dict):
        if bundle is None:
            raise HTTPException(status_code=503, detail="Train the local research model first")
        features = bundle["features"]
        row = payload.get("features")
        if not isinstance(row, dict):
            raise HTTPException(status_code=422, detail="Provide an object at features")
        missing = sorted(set(features) - set(row))
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing model features: {missing[:8]}")
        import pandas as pd
        X = pd.DataFrame([{k: row[k] for k in features}], columns=features)
        raw = float(bundle["model"].predict_proba(X)[0, 1])
        import numpy as np
        eps = 1e-6
        logit = np.log(np.clip(raw, eps, 1 - eps) / (1 - np.clip(raw, eps, 1 - eps)))
        probability = float(bundle["calibrator"].predict_proba([[logit]])[0, 1])
        return {"risk_score": probability, "review_flag_at_research_threshold": probability >= bundle["threshold"],
                "disclaimer": "Not for real credit decisions; threshold reflects illustrative costs."}

    return app


app = create_app()
