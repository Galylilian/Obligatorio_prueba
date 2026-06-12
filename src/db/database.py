from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from src.settings.config import DATABASE_URL

# =============================
# ENGINE Y SESIÓN
# =============================
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """
    Dependency para FastAPI.
    Provee una sesión de DB por request y la cierra al terminar.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Crea todas las tablas si no existen.
    Se llama al iniciar la API.
    """
    from src.db import models  # noqa: F401 — necesario para que SQLAlchemy registre los modelos
    Base.metadata.create_all(bind=engine)