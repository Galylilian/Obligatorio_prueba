
import io
from PIL import Image


def _dummy_image_bytes():
    """Genera una imagen dummy en memoria para usar en tests."""
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
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
def test_predict_returns_label_and_confidence(client):
    """El endpoint /predict debe devolver label y confidence."""
    img_bytes = _dummy_image_bytes()

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200

    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert data["label"] in ["fall", "no_fall"]
    assert 0.0 <= data["confidence"] <= 1.0


# =============================
# GRADCAM
# =============================
def test_gradcam_returns_image(client):
    """El endpoint /gradcam debe devolver una imagen JPEG."""
    img_bytes = _dummy_image_bytes()

    response = client.post(
        "/gradcam",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
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