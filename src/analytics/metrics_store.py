import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from src.analytics.db import analytics_enabled, get_session
from src.analytics.models import ModelMetric
from src.settings.config import settings

METRICS_JSON_PATH = Path(settings.model_path).parent / "evaluation_metrics.json"


def metrics_payload(
    *,
    model_name: str,
    split: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float,
    evaluated_at: datetime | None = None,
) -> dict:
    return {
        "model_name": model_name,
        "split": split,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "evaluated_at": (evaluated_at or datetime.now(timezone.utc)).isoformat(),
    }


def save_metrics_json(splits: dict[str, dict], model_name: str = "resnet18") -> Path:
    evaluated_at = datetime.now(timezone.utc)
    payload = {
        "model_name": model_name,
        "evaluated_at": evaluated_at.isoformat(),
        "splits": {
            split: metrics_payload(
                model_name=model_name,
                split=split,
                evaluated_at=evaluated_at,
                **values,
            )
            for split, values in splits.items()
        },
    }
    METRICS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return METRICS_JSON_PATH


def persist_model_metrics(splits: dict[str, dict], model_name: str = "resnet18") -> None:
    save_metrics_json(splits, model_name=model_name)
    if not analytics_enabled():
        return

    evaluated_at = datetime.now(timezone.utc)
    with get_session() as session:
        for split, values in splits.items():
            session.add(
                ModelMetric(
                    model_name=model_name,
                    split=split,
                    accuracy=float(values["accuracy"]),
                    precision=float(values["precision"]),
                    recall=float(values["recall"]),
                    f1_score=float(values["f1_score"]),
                    evaluated_at=evaluated_at,
                )
            )


def get_latest_model_metrics() -> dict:
    if not analytics_enabled():
        return {"model_metrics_enabled": False, "splits": {}}

    with get_session() as session:
        has_rows = session.scalar(select(ModelMetric.id).limit(1)) is not None

    if not has_rows and METRICS_JSON_PATH.exists():
        seed_model_metrics_from_file()

    with get_session() as session:
        rows = session.scalars(
            select(ModelMetric).order_by(ModelMetric.evaluated_at.desc())
        ).all()

        latest_by_split: dict[str, ModelMetric] = {}
        for row in rows:
            if row.split not in latest_by_split:
                latest_by_split[row.split] = row

        return {
            "model_metrics_enabled": bool(latest_by_split),
            "model_name": next(iter(latest_by_split.values())).model_name if latest_by_split else None,
            "evaluated_at": next(iter(latest_by_split.values())).evaluated_at.isoformat()
            if latest_by_split
            else None,
            "splits": {
                split: {
                    "accuracy": row.accuracy,
                    "precision": row.precision,
                    "recall": row.recall,
                    "f1_score": row.f1_score,
                    "evaluated_at": row.evaluated_at.isoformat(),
                }
                for split, row in latest_by_split.items()
            },
        }


def seed_model_metrics_from_file() -> bool:
    if not METRICS_JSON_PATH.exists() or not analytics_enabled():
        return False

    payload = json.loads(METRICS_JSON_PATH.read_text(encoding="utf-8"))
    splits = {
        split: {
            "accuracy": values["accuracy"],
            "precision": values["precision"],
            "recall": values["recall"],
            "f1_score": values["f1_score"],
        }
        for split, values in payload.get("splits", {}).items()
    }
    if not splits:
        return False

    with get_session() as session:
        existing = session.scalar(select(ModelMetric.id).limit(1))
        if existing is not None:
            return False

    persist_model_metrics(splits, model_name=payload.get("model_name", "resnet18"))
    return True
