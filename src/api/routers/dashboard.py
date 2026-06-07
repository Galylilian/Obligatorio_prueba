from fastapi import APIRouter

from src.analytics.metrics_store import get_latest_model_metrics
from src.analytics.repository import get_dashboard_stats

router = APIRouter(prefix="/dashboard", tags=["Analytics"])


@router.get("/stats")
def dashboard_stats():
    """KPIs operacionales + metricas del modelo."""
    stats = get_dashboard_stats()
    stats["model"] = get_latest_model_metrics()
    return stats


@router.get("/model-metrics")
def model_metrics():
    """Accuracy, F1, precision y recall del ultimo entrenamiento."""
    return get_latest_model_metrics()
