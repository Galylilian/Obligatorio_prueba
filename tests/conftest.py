import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.app import app
from src.db.database import get_db, Base

# =============================
# DB SQLITE (ARCHIVO LOCAL) PARA TESTS ✅
# =============================
SQLALCHEMY_TEST_URL = "sqlite:///./test.db"

engine_test = create_engine(
    SQLALCHEMY_TEST_URL,
    connect_args={"check_same_thread": False}
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# =============================
# FIXTURE PRINCIPAL ✅
# =============================
@pytest.fixture
def client():
    # crear tablas en DB de test
    Base.metadata.create_all(bind=engine_test)

    # reemplazar DB real por DB de test
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    # limpiar después del test
    Base.metadata.drop_all(bind=engine_test)
    app.dependency_overrides.clear()