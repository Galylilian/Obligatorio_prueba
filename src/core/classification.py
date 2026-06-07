import torch
from PIL import Image
from typing import List
import time

from src.core.model import get_model
from src.core.preprocessing.transforms import get_test_transforms
from src.settings.config import DEVICE, MODEL_PATH


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

        self.classes = ["fall", "no_fall"]

    def predict(self, images: List[Image.Image]):
        start = time.time()
        results = []

        for img in images:
            tensor = self.transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            results.append({
                "label": self.classes[pred],
                "confidence": float(probs[0][pred])
            })

        return {
            "images": results,
            "inference_time": round(time.time() - start, 4)
        }

