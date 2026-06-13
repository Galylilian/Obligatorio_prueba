"""
scripts/convert_dataset.py

Convierte las imágenes scrapeadas a formato de clasificación binaria:
    data/processed/train/fall/
    data/processed/train/no_fall/
    data/processed/valid/fall/
    data/processed/valid/no_fall/
    data/processed/test/fall/
    data/processed/test/no_fall/

Fuente única:
    data/raw/scraped/fall/     → imágenes de caídas
    data/raw/scraped/no_fall/  → imágenes sin caída

Las imágenes se dividen aleatoriamente en 70% train / 15% valid / 15% test.
La semilla es fija (42) para garantizar reproducibilidad.

Al finalizar genera:
    data/processed/dataset_labels.csv  → trazabilidad completa de cada imagen
    (filename, label, source, query, split, timestamp)

Correr primero:
    python scripts/scrape_dataset.py
"""

import os
import csv
import shutil
import random
import pathlib
from datetime import datetime

# =============================
# PATHS
# =============================
ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRAPED_PATH = ROOT / "data" / "raw" / "scraped"
SCRAPING_LOG = SCRAPED_PATH / "scraping_log.csv"
OUTPUT_PATH = ROOT / "data" / "processed"
LABELS_CSV = OUTPUT_PATH / "dataset_labels.csv"

SPLITS = ["train", "valid", "test"]
LABELS = ["fall", "no_fall"]

# División del dataset — suma 1.0
SPLIT_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}

# Semilla fija para reproducibilidad del split
RANDOM_SEED = 42


# =============================
# UTILIDADES
# =============================

def is_image(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))


def split_list(items: list, ratios: dict, seed: int) -> dict:
    """
    Divide una lista en train/valid/test según los ratios dados.
    La semilla garantiza que el mismo dataset siempre produce el mismo split.
    """
    random.seed(seed)
    shuffled = items.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_valid = int(n * ratios["valid"])

    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_valid],
        "test":  shuffled[n_train + n_valid:],
    }


# =============================
# VALIDAR ENTRADA
# =============================
if not SCRAPED_PATH.exists():
    print("❌ No existe data/raw/scraped/")
    print("   Corré primero: python scripts/scrape_dataset.py")
    exit(1)

# =============================
# LIMPIAR Y CREAR OUTPUT
# =============================
if OUTPUT_PATH.exists():
    shutil.rmtree(OUTPUT_PATH)

for split in SPLITS:
    for label in LABELS:
        (OUTPUT_PATH / split / label).mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print("CONVERT DATASET — Scraped images → train/valid/test")
print("=" * 60)

# =============================
# LEER LOG DE SCRAPING
# Para recuperar la query de origen de cada imagen
# =============================
scraping_queries = {}
if SCRAPING_LOG.exists():
    with open(SCRAPING_LOG, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scraping_queries[row["filename"]] = row.get("query", "")

# =============================
# PROCESAR IMÁGENES POR CLASE
# =============================
label_rows = []

for label in LABELS:

    src_dir = SCRAPED_PATH / label

    if not src_dir.exists():
        print(f"\n⚠️  No existe {src_dir} — saltando")
        continue

    files = [f for f in os.listdir(src_dir) if is_image(f)]

    if not files:
        print(f"\n⚠️  {label}: sin imágenes")
        continue

    print(f"\n📁 {label.upper()}: {len(files)} imágenes → split {SPLIT_RATIOS}")

    splits_assigned = split_list(files, SPLIT_RATIOS, RANDOM_SEED)

    for split, split_files in splits_assigned.items():

        dest_dir = OUTPUT_PATH / split / label
        print(f"  [{split}]: {len(split_files)} imágenes")

        for file in split_files:

            src = src_dir / file
            dst = dest_dir / file
            shutil.copy(str(src), str(dst))

            label_rows.append({
                "filename": file,
                "label": label,
                "source": "duckduckgo",
                "query": scraping_queries.get(file, ""),
                "split": split,
                "timestamp": datetime.now().isoformat(),
            })

# =============================
# GENERAR CSV DE TRAZABILIDAD
# =============================
print("\n" + "=" * 60)
print("Generando dataset_labels.csv")
print("=" * 60)

with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["filename", "label", "source", "query", "split", "timestamp"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(label_rows)

print(f"  ✅ CSV guardado en: {LABELS_CSV}")
print(f"  📊 Total registradas: {len(label_rows)} imágenes")

# =============================
# RESUMEN FINAL
# =============================
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)

for split in SPLITS:
    fall_count    = len([f for f in os.listdir(OUTPUT_PATH / split / "fall")    if is_image(f)])
    no_fall_count = len([f for f in os.listdir(OUTPUT_PATH / split / "no_fall") if is_image(f)])
    total = fall_count + no_fall_count
    print(f"  [{split}] fall: {fall_count} | no_fall: {no_fall_count} | total: {total}")

print("\n✅ Dataset listo en data/processed/")
print(f"✅ Trazabilidad guardada en {LABELS_CSV}")