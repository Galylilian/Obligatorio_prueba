from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_docs_available():
    """Verifica que la documentación Swagger está disponible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_health():
    """Verifica que el endpoint de health responde correctamente."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}