"""Predicción batch offline sobre carpeta de imágenes."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.model import get_model  # noqa: E402
from src.preprocessing.transforms import get_eval_transforms  # noqa: E402
from src.settings.config import DEVICE, LABEL_ENCODER_PATH, MODEL_PATH  # noqa: E402
from src.utils.label_encoder import decode_prediction, load_label_encoder  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def predict_batch(image_paths: list[Path], model, transform, label_encoder) -> pd.DataFrame:
    rows = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_idx = model(tensor).argmax().item()
        rows.append(
            {
                "path": str(path.resolve()),
                "filename": path.name,
                "prediction": pred_idx,
                "label": decode_prediction(label_encoder, pred_idx),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicción batch Fall / Not Fall")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Carpeta con imágenes (recursivo)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/metadata/batch_predictions.csv"),
        help="CSV de salida",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {args.input_dir}")

    image_paths = collect_images(args.input_dir)
    if not image_paths:
        raise ValueError(f"No se encontraron imágenes en {args.input_dir}")

    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    label_encoder = load_label_encoder(LABEL_ENCODER_PATH)
    transform = get_eval_transforms()

    df = predict_batch(image_paths, model, transform, label_encoder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"{len(df)} predicciones guardadas en {args.output}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
