from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.analytics.models import Base
from src.settings.config import settings

_engine = None
_SessionLocal = None


def analytics_enabled() -> bool:
    return bool(settings.database_url)


def get_engine():
    global _engine, _SessionLocal
    if not analytics_enabled():
        return None
    if _engine is None:
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def init_db() -> None:
    if not analytics_enabled():
        return
    try:
        engine = get_engine()
        if engine is not None:
            Base.metadata.create_all(bind=engine)
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
