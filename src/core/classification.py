import torch
from PIL import Image
from typing import List
import time

from src.core.model import get_model
from src.core.detector import get_person_detector
from src.core.preprocessing.cropping import crop_to_box
from src.core.preprocessing.transforms import get_test_transforms
from src.settings.config import DEVICE, MODEL_PATH

CROP_MARGIN = 0.15


class ImageClassifier:
    def __init__(self):
        # ✅ IMPORTANTE: pretrained=False
        self.model = get_model(pretrained=False)

        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE)
        )

        self.model.to(DEVICE)
        self.model.eval()

        self.transform = get_test_transforms()

        # Detector de personas (paso previo a la clasificacion): recorta al
        # bounding box de la persona con el mismo margen que convert_dataset.py
        # usa sobre el box dibujado a mano, para no reintroducir skew entre
        # entrenamiento y produccion.
        self.detector = get_person_detector()

        self.classes = ["fall", "no_fall"]

    def predict(self, images: List[Image.Image]):
        start = time.time()
        results = []

        for img in images:
            box = self.detector.detect(img)

            # Sin persona detectada: no se fuerza una clase, se devuelve
            # explicitamente que no hay nada que clasificar.
            if box is None:
                results.append({
                    "label": None,
                    "confidence": None,
                    "person_detected": False,
                    "bbox": None,
                })
                continue

            crop = crop_to_box(img, box, margin=CROP_MARGIN)
            tensor = self.transform(crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            results.append({
                "label": self.classes[pred],
                "confidence": float(probs[0][pred]),
                "person_detected": True,
                "bbox": box,
            })

        return {
            "images": results,
            "inference_time": round(time.time() - start, 4)
        }
