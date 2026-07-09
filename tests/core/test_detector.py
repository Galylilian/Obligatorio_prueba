from pathlib import Path

from PIL import Image

from src.core.detector import PersonDetector

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_detect_returns_none_without_person():
    """Una imagen gris solida no tiene ninguna persona: debe devolver None."""
    detector = PersonDetector(device="cpu")
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))

    assert detector.detect(img) is None


def test_detect_finds_person_in_real_photo():
    """Sobre una foto real con una persona, debe devolver un box valido."""
    detector = PersonDetector(device="cpu")
    img = Image.open(FIXTURES_DIR / "person.jpg").convert("RGB")

    box = detector.detect(img)

    assert box is not None
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= 1
    assert 0 <= y1 < y2 <= 1
