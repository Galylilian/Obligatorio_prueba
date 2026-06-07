from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from src.analytics.db import analytics_enabled, get_session
from src.analytics.models import PredictionEvent
from src.settings.config import settings


def record_prediction(
    *,
    label: str,
    confidence: float,
    person_id: str | None = None,
    source: str | None = None,
) -> None:
    if not analytics_enabled():
        return
    with get_session() as session:
        session.add(
            PredictionEvent(
                person_id=person_id,
                label=label,
                confidence=confidence,
                source=source,
            )
        )


def get_dashboard_stats() -> dict:
    if not analytics_enabled():
        return {
            "falls_today": 0,
            "falls_week": 0,
            "high_risk_persons": 0,
            "analytics_enabled": False,
        }

    now = datetime.now(timezone.utc)
    start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_week = now - timedelta(days=7)
    min_falls = settings.high_risk_min_falls_week

    with get_session() as session:
        falls_today = session.scalar(
            select(func.count())
            .select_from(PredictionEvent)
            .where(
                PredictionEvent.label == "fall",
                PredictionEvent.created_at >= start_today,
            )
        ) or 0

        falls_week = session.scalar(
            select(func.count())
            .select_from(PredictionEvent)
            .where(
                PredictionEvent.label == "fall",
                PredictionEvent.created_at >= start_week,
            )
        ) or 0

        risky_subq = (
            select(PredictionEvent.person_id)
            .where(
                PredictionEvent.label == "fall",
                PredictionEvent.created_at >= start_week,
                PredictionEvent.person_id.is_not(None),
            )
            .group_by(PredictionEvent.person_id)
            .having(func.count() >= min_falls)
            .subquery()
        )
        high_risk_persons = session.scalar(
            select(func.count()).select_from(risky_subq)
        ) or 0

    return {
        "falls_today": falls_today,
        "falls_week": falls_week,
        "high_risk_persons": high_risk_persons,
        "analytics_enabled": True,
    }
