from io import BytesIO

from PIL import Image


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_requires_file(client):
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_with_image(client):
    img = Image.new("RGB", (224, 224), color="red")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert data["label"] in ("fall", "not_fall")
