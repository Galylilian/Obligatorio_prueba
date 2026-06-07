from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.analytics.models import Base
from src.settings.config import settings

_engine = None
_SessionLocal = None


def analytics_enabled() -> bool:
    return bool(settings.effective_database_url)


def get_engine():
    global _engine, _SessionLocal
    if not analytics_enabled():
        return None
    if _engine is None:
        url = settings.effective_database_url
        if url.startswith("sqlite"):
            db_path = url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(
                url,
                connect_args={"check_same_thread": False},
            )
        else:
            _engine = create_engine(url, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def init_db() -> None:
    if not analytics_enabled():
        return
    try:
        engine = get_engine()
        if engine is not None:
            Base.metadata.create_all(bind=engine)
            from src.analytics.metrics_store import seed_model_metrics_from_file

            seed_model_metrics_from_file()
    except Exception as exc:
        from src.utils.logger import get_logger

        get_logger("analytics").warning(f"No se pudo inicializar PostgreSQL: {exc}")


@contextmanager
def get_session() -> Session:
    get_engine()
    if _SessionLocal is None:
        raise RuntimeError("Analytics DB no configurada (DATABASE_URL vacia)")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
