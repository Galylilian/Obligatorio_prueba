
import io
from pathlib import Path

from PIL import Image

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _dummy_image_bytes():
    """Imagen gris solida en memoria: no contiene ninguna persona real, asi
    que el detector (PersonDetector) no deberia encontrar nada."""
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _person_image_bytes():
    """Foto real con una persona (tests/fixtures/person.jpg), para ejercitar
    el camino donde el detector SI encuentra a alguien."""
    buf = io.BytesIO((FIXTURES_DIR / "person.jpg").read_bytes())
    buf.seek(0)
    return buf


# =============================
# HEALTH
# =============================
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# =============================
# PREDICT
# =============================
def test_predict_without_person_returns_no_detection(client):
    """Sin ninguna persona en la imagen, /predict no debe forzar una clase."""
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", _dummy_image_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200

    data = response.json()
    assert data["person_detected"] is False
    assert data["label"] is None
    assert data["confidence"] is None


def test_predict_with_person_returns_label_and_confidence(client):
    """Con una persona real en la imagen, /predict debe devolver label, confidence y bbox."""
    response = client.post(
        "/predict",
        files={"file": ("person.jpg", _person_image_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200

    data = response.json()
    assert data["person_detected"] is True
    assert data["label"] in ["fall", "no_fall"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["bbox"]) == 4


# =============================
# GRADCAM
# =============================
def test_gradcam_without_person_returns_error(client):
    """Sin ninguna persona detectada, /gradcam no puede generar un heatmap."""
    response = client.post(
        "/gradcam",
        files={"file": ("test.jpg", _dummy_image_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "error" in data


def test_gradcam_with_person_returns_image(client):
    """Con una persona real, /gradcam debe devolver una imagen JPEG."""
    response = client.post(
        "/gradcam",
        files={"file": ("person.jpg", _person_image_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200
    assert "image" in response.headers.get("content-type", "")


# =============================
# DASHBOARD
# =============================
def test_dashboard_stats_structure(client):
    """El endpoint /dashboard/stats debe devolver la estructura correcta."""
    response = client.get("/dashboard/stats")

    assert response.status_code == 200

    data = response.json()
    assert "total_predictions" in data
    assert "falls_today" in data
    assert "falls_week" in data
    assert "classified_today" in data
    assert "model" in data
