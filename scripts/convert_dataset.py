"""
scripts/convert_dataset.py

PASO 3: Arma el dataset final a partir de las carpetas etiquetadas.

Lee las imagenes desde:
    data/raw/fall/      (imagenes de caidas)
    data/raw/no_fall/   (imagenes de no caidas)

Y las divide en:
    data/processed/train/fall/
    data/processed/train/no_fall/
    data/processed/valid/...
    data/processed/test/...

Division: 70% train / 15% valid / 15% test, estratificada por clase.
Semilla fija (42) para reproducibilidad.

Genera data/processed/dataset_labels.csv con trazabilidad completa.

Uso:
    python scripts/convert_dataset.py
"""

import csv
import pathlib
import random
import shutil
from collections import defaultdict
from datetime import datetime

ROOT         = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR      = ROOT / "data" / "raw"
FALL_DIR     = RAW_DIR / "fall"
NO_FALL_DIR  = RAW_DIR / "no_fall"
POOL_LOG     = RAW_DIR / "pool" / "pool_log.csv"
OUTPUT       = ROOT / "data" / "processed"
LABELS_OUT   = OUTPUT / "dataset_labels.csv"

SPLITS       = ["train", "valid", "test"]
LABELS       = ["fall", "no_fall"]
SPLIT_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}
RANDOM_SEED  = 42
IMG_EXTS     = {".jpg", ".jpeg", ".png"}


def split_list(items: list, ratios: dict, seed: int) -> dict:
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_valid  = int(n * ratios["valid"])
    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train : n_train + n_valid],
        "test":  shuffled[n_train + n_valid :],
    }


def collect_images(folder: pathlib.Path) -> list[pathlib.Path]:
    if not folder.exists():
        return []
    return [f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS]


def load_source_lookup() -> dict[str, str]:
    """filename -> procedencia real (pexels/video), segun pool_log.csv"""
    if not POOL_LOG.exists():
        return {}
    with open(POOL_LOG, encoding="utf-8") as f:
        return {row["filename"]: row["source"] for row in csv.DictReader(f)}


print("\n" + "=" * 60)
print("PASO 3: ARMADO DEL DATASET FINAL")
print("=" * 60)

by_label: dict[str, list[pathlib.Path]] = {
    "fall":    collect_images(FALL_DIR),
    "no_fall": collect_images(NO_FALL_DIR),
}

print("\nImagenes encontradas:")
for label in LABELS:
    print(f"  {label}: {len(by_label[label])}")

if not by_label["fall"]:
    print(f"\nERROR: No hay imagenes en {FALL_DIR}")
    print("  Coloca imagenes de caidas ahi o usa label_tool.py")
    exit(1)

if not by_label["no_fall"]:
    print(f"\nERROR: No hay imagenes en {NO_FALL_DIR}")
    print("  Coloca imagenes de no caidas ahi o usa label_tool.py")
    exit(1)

# Limpiar y recrear estructura de salida
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)
for split in SPLITS:
    for label in LABELS:
        (OUTPUT / split / label).mkdir(parents=True, exist_ok=True)

source_lookup = load_source_lookup()
label_rows: list[dict] = []

for label in LABELS:
    files = by_label[label]
    splits = split_list(files, SPLIT_RATIOS, RANDOM_SEED)

    print(f"\n  {label}:")
    for split, split_files in splits.items():
        dest_dir = OUTPUT / split / label
        print(f"    [{split}]: {len(split_files)}")

        for src in split_files:
            dst = dest_dir / src.name
            shutil.copy(str(src), str(dst))

            label_rows.append({
                "filename":  src.name,
                "label":     label,
                "source":    source_lookup.get(src.name, "unknown"),
                "split":     split,
                "timestamp": datetime.now().isoformat(),
            })

with open(LABELS_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "label", "source", "split", "timestamp"])
    writer.writeheader()
    writer.writerows(label_rows)

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
for split in SPLITS:
    fall    = len(list((OUTPUT / split / "fall").glob("*")))
    no_fall = len(list((OUTPUT / split / "no_fall").glob("*")))
    print(f"  [{split}] fall: {fall} | no_fall: {no_fall} | total: {fall + no_fall}")

print(f"\nDataset listo en      : {OUTPUT}")
print(f"Trazabilidad en       : {LABELS_OUT}")
print(f"Total de imagenes     : {len(label_rows)}")
