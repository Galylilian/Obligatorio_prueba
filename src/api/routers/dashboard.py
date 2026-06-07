from fastapi import APIRouter

from src.analytics.repository import get_dashboard_stats

router = APIRouter(prefix="/dashboard", tags=["Analytics"])


@router.get("/stats")
def dashboard_stats():
    """KPIs para Grafana, Superset o Power BI."""
    return get_dashboard_stats()
