from PIL import Image

from src.core.preprocessing.cropping import crop_to_box


def test_crop_to_box_no_margin():
    """Sin margen, el recorte debe coincidir exactamente con el box (en pixeles)."""
    img = Image.new("RGB", (200, 100))
    cropped = crop_to_box(img, (0.25, 0.5, 0.75, 1.0), margin=0.0)

    assert cropped.size == (100, 50)


def test_crop_to_box_applies_margin():
    """Con margen, el recorte debe ser mas grande que el box original."""
    img = Image.new("RGB", (200, 200))
    tight = crop_to_box(img, (0.4, 0.4, 0.6, 0.6), margin=0.0)
    padded = crop_to_box(img, (0.4, 0.4, 0.6, 0.6), margin=0.15)

    assert padded.size[0] > tight.size[0]
    assert padded.size[1] > tight.size[1]


def test_crop_to_box_clamps_at_image_edges():
    """El margen no debe salirse de los bordes de la imagen."""
    img = Image.new("RGB", (100, 100))
    # Box ya pegado al borde: expandir con margen no debe crashear ni salirse.
    cropped = crop_to_box(img, (0.0, 0.0, 0.2, 0.2), margin=0.5)

    assert cropped.size[0] <= 100
    assert cropped.size[1] <= 100
