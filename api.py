# noshow_iq/api.py
# FastAPI service – exposes /health, /predict, /history, /stats.

import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import model, preprocess, db

app = FastAPI(title='NoShowIQ', version='1.0')


# ── Request schema ──────────────────────────────────────────────────────────

class Appointment(BaseModel):
    Gender:          str
    Age:             int = Field(ge=0, le=110)
    scheduled_day:   str   # ISO 8601 datetime string
    appointment_day: str
    Scholarship:     int = 0
    hypertension:    int = 0
    Diabetes:        int = 0
    Alcoholism:      int = 0
    handicap:        int = 0
    sms_received:    int = 0


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    """Liveness check – always returns 200."""
    return {'status': 'ok', 'time': datetime.now(timezone.utc).isoformat()}


@app.post('/predict')
def predict_endpoint(appt: Appointment):
    """Run the ML pipeline and return risk level + recommendation."""
    try:
        import pandas as pd
        raw = appt.model_dump()
        df  = pd.DataFrame([raw])
        df  = preprocess.engineer(preprocess.clean(df))
        result = model.predict(df.iloc[0].to_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    db.log_prediction(
        raw_input=raw,
        cleaned=df.iloc[0].to_dict(),
        result=result,
    )
    return result


@app.get('/history')
def history():
    """Return the last 20 predictions, newest first."""
    return db.last_n_predictions(20)


@app.get('/stats')
def stats():
    """Return aggregated statistics computed entirely inside MongoDB."""
    return db.aggregate_stats()
