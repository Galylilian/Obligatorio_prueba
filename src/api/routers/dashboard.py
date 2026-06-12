from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

from src.db.database import get_db
from src.db.models import Prediction

router = APIRouter()

# ✅ ruta absoluta para evitar problemas en Docker
BASE_DIR = Path(__file__).resolve().parents[3]
METRICS_FILE = BASE_DIR / "metrics.json"


@router.get("/dashboard/stats")
def get_stats(db: Session = Depends(get_db)):

    # =============================
    # STATS DESDE BASE DE DATOS ✅
    # =============================
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = now - timedelta(days=7)

    # total de predicciones
    total_predictions = db.query(func.count(Prediction.id)).scalar() or 0

    # caídas detectadas hoy
    falls_today = db.query(func.count(Prediction.id)).filter(
        Prediction.label == "fall",
        Prediction.created_at >= today_start
    ).scalar() or 0

    # caídas detectadas en la semana
    falls_week = db.query(func.count(Prediction.id)).filter(
        Prediction.label == "fall",
        Prediction.created_at >= week_start
    ).scalar() or 0

    # total de imágenes clasificadas hoy
    classified_today = db.query(func.count(Prediction.id)).filter(
        Prediction.created_at >= today_start
    ).scalar() or 0

    # distribución de labels (total histórico)
    label_counts = db.query(
        Prediction.label,
        func.count(Prediction.id)
    ).group_by(Prediction.label).all()

    label_distribution = {label: count for label, count in label_counts}

    # =============================
    # MÉTRICAS DEL MODELO (metrics.json) ✅
    # =============================
    model_metrics = {}

    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE) as f:
                model_metrics = json.load(f)
        except Exception:
            model_metrics = {}

    # =============================
    # RESPUESTA PARA STREAMLIT Y GRAFANA ✅
    # =============================
    return {
        # 🔹 contadores operacionales (desde DB)
        "total_predictions": total_predictions,
        "classified_today": classified_today,
        "falls_today": falls_today,
        "falls_week": falls_week,
        "label_distribution": label_distribution,

        # compatibilidad con Streamlit existente
        "high_risk_persons": falls_today,
        "analytics_enabled": True,

        # ✅ estructura que espera Streamlit para métricas del modelo
        "model": {
            "splits": {
                "valid": model_metrics
            }
        }
    }
