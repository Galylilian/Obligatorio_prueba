"""
Compara las predicciones del modelo normal (resnet18.pth) contra el modelo
cuantizado (resnet18_quantized.pth) sobre el conjunto de test real
(data/processed/test/), imagen por imagen.

A diferencia de benchmark_quantization.py (que mide latencia), este script
mide si ambos modelos coinciden en la clase predicha y compara su confianza,
para detectar si la cuantización dinámica degrada las predicciones.
"""

import json
import pathlib

import torch

from src.core.model import get_model
from src.core.preprocessing.transforms import get_test_transforms
from src.data.dataset import get_dataloaders
from src.utils.logger import get_logger

logger = get_logger("compare_models")

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "resnet18.pth"
QUANTIZED_MODEL_PATH = MODELS_DIR / "resnet18_quantized.pth"

CLASSES = ["fall", "no_fall"]


def load_normal_model():
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def load_quantized_model():
    model = get_model(pretrained=False)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    quantized_model.load_state_dict(torch.load(QUANTIZED_MODEL_PATH, map_location="cpu"))
    quantized_model.eval()
    return quantized_model


def predict(model, tensor):
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return CLASSES[pred], float(probs[0][pred])


def main():
    _, _, test_loader = get_dataloaders(batch_size=1)
    filepaths = [path for path, _ in test_loader.dataset.samples]

    logger.info("Cargando modelo normal y modelo cuantizado...")
    normal_model = load_normal_model()
    quantized_model = load_quantized_model()

    results = []
    agreements = 0

    for (image, _), filepath in zip(test_loader, filepaths):
        filename = pathlib.Path(filepath).name

        normal_label, normal_conf = predict(normal_model, image)
        quantized_label, quantized_conf = predict(quantized_model, image)

        agree = normal_label == quantized_label
        agreements += int(agree)

        results.append({
            "image": filename,
            "normal": {"label": normal_label, "confidence": round(normal_conf, 4)},
            "quantized": {"label": quantized_label, "confidence": round(quantized_conf, 4)},
            "agree": agree,
        })

        marker = "✅" if agree else "⚠️"
        logger.info(
            f"{marker} {filename} -> normal: {normal_label} ({normal_conf:.4f}) | "
            f"quantized: {quantized_label} ({quantized_conf:.4f})"
        )

    total = len(results)
    agreement_pct = (agreements / total * 100) if total else 0.0

    logger.info(
        f"Acuerdo entre modelos: {agreements}/{total} ({agreement_pct:.1f}%)"
    )

    disagreements = [r for r in results if not r["agree"]]
    if disagreements:
        logger.info("Imágenes con predicción distinta entre normal y cuantizado:")
        for r in disagreements:
            logger.info(f"  - {r['image']}: normal={r['normal']['label']} vs quantized={r['quantized']['label']}")

    output_path = BASE_DIR / "compare_models.json"
    with open(output_path, "w") as f:
        json.dump({
            "total": total,
            "agreements": agreements,
            "agreement_pct": agreement_pct,
            "results": results,
        }, f, indent=2)

    logger.info(f"✅ Resultados guardados en: {output_path}")


if __name__ == "__main__":
    main()
