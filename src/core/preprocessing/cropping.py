from PIL import Image

BoxRatio = tuple[float, float, float, float]


def crop_to_box(image: Image.Image, box_ratio: BoxRatio, margin: float = 0.15) -> Image.Image:
    """
    Recorta `image` al bounding box `box_ratio` (x1, y1, x2, y2 normalizados 0-1),
    expandido un `margin` (fraccion del ancho/alto del box) en cada direccion para
    no cortar pies/manos/cabeza, clamped a los bordes de la imagen.

    Usado tanto para curar el dataset de entrenamiento (convert_dataset.py) como
    para el recorte en produccion (PersonDetector), para que ambos vean exactamente
    el mismo preprocesamiento.
    """
    x1, y1, x2, y2 = box_ratio
    box_w, box_h = x2 - x1, y2 - y1

    x1 = max(0.0, x1 - box_w * margin)
    y1 = max(0.0, y1 - box_h * margin)
    x2 = min(1.0, x2 + box_w * margin)
    y2 = min(1.0, y2 + box_h * margin)

    width, height = image.size
    return image.crop((
        round(x1 * width),
        round(y1 * height),
        round(x2 * width),
        round(y2 * height),
    ))
