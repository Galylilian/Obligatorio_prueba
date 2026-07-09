from PIL import Image
import torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)

# Indice de "person" en las 91 clases de COCO (0 es background).
COCO_PERSON_LABEL = 1

BoxRatio = tuple[float, float, float, float]


class PersonDetector:
    """
    Detector de personas (COCO-pretrained, sin fine-tuning) usado como paso
    previo a la clasificacion fall/no_fall, tanto en produccion (API) como
    podria usarse offline. Devuelve el bounding box de mayor confianza como
    ratios normalizados 0-1, para que sea directamente comparable con los
    boxes dibujados a mano en label_tool.py (ver bbox_log.csv).

    ssdlite320_mobilenet_v3_large se eligio por ser el modelo de deteccion
    mas liviano de torchvision: el proyecto corre inferencia en CPU (EC2 sin
    GPU), asi que se prioriza latencia sobre precision de localizacion.
    """

    def __init__(self, device: str = "cpu", score_threshold: float = 0.5):
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=weights)
        self.model.to(device)
        self.model.eval()

        self.transform = weights.transforms()
        self.device = device
        self.score_threshold = score_threshold

    def detect(self, image: Image.Image) -> BoxRatio | None:
        tensor = self.transform(image).to(self.device)

        with torch.no_grad():
            output = self.model([tensor])[0]

        width, height = image.size
        best_box = None
        best_score = 0.0

        for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
            if label.item() != COCO_PERSON_LABEL or score.item() < self.score_threshold:
                continue
            if score.item() > best_score:
                best_score = score.item()
                x1, y1, x2, y2 = box.tolist()
                best_box = (x1 / width, y1 / height, x2 / width, y2 / height)

        return best_box


_detector: PersonDetector | None = None


def get_person_detector() -> PersonDetector:
    """Singleton compartido entre classification.py y el router de gradcam,
    para no cargar el modelo de deteccion dos veces."""
    global _detector
    if _detector is None:
        from src.settings.config import DEVICE
        _detector = PersonDetector(device=DEVICE)
    return _detector
