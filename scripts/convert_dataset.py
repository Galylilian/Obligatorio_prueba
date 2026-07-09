"""
scripts/convert_dataset.py

PASO 3: Arma el dataset final a partir de las imagenes etiquetadas.

Una imagen puede tener MAS DE UNA PERSONA, cada una con su propio label
(fall/no_fall) y su propio bounding box: la unidad de etiquetado no es la
imagen, es la PERSONA. Por eso la fuente de verdad es data/raw/bbox_log.csv
(una fila por persona: filename, label, x1, y1, x2, y2), no una carpeta
fall/ o no_fall/ — una misma imagen puede aportar personas de ambas clases.

Lee las imagenes fuente desde data/raw/labeled/ (ahi las deja label_tool.py
cuando confirmas "Listo, siguiente" en una imagen), y por cada fila de
bbox_log.csv recorta esa imagen al bounding box correspondiente (con margen,
ver src/core/preprocessing/cropping.py) — el mismo recorte que aplica el
detector de personas en produccion (src/core/detector.py), para no
reintroducir skew entre entrenamiento y serving. Cada persona se convierte
asi en un ejemplo de entrenamiento independiente, aunque varias vengan de la
misma imagen original.

Toda fila de bbox_log.csv debe tener procedencia conocida en
data/raw/pool/pool_log.csv (source=pexels o source=video), y su imagen debe
estar en data/raw/labeled/ (osea, "Listo" ya fue confirmado para ella). El
script se corta si encuentra alguna fila que no cumpla esto.

Y las divide en:
    data/processed/train/fall/
    data/processed/train/no_fall/
    data/processed/valid/...
    data/processed/test/...

Division: 70% train / 15% valid / 15% test, estratificada por (clase, fuente):
cada combinacion label x source (fall/pexels, fall/video, no_fall/pexels,
no_fall/video) se divide por separado con las mismas proporciones antes de
combinar los splits. Esto evita que train/valid/test tengan una mezcla de
fuentes distinta dentro de cada clase, lo que haria que el modelo aprenda a
reconocer "pinta de video" o "pinta de foto de stock" en vez de la caida en si.

La unidad que se estratifica NO es la imagen individual, es su "grupo de
casi-duplicados": frames de video separados por poco tiempo (o fotos casi
identicas) se agrupan primero via un perceptual hash (dHash) y TODO el grupo
va al mismo split. Sin esto, dos frames casi identicos de un mismo momento
del video podrian terminar uno en train y otro en test — el modelo
"reconoceria" el frame de test por haberlo visto casi igual en train,
inflando la accuracy reportada sin que el modelo generalice mejor de verdad
(hallazgo del EDA en notebooks/eda.ipynb).

Semilla fija (42) para reproducibilidad.

Genera data/processed/dataset_labels.csv con trazabilidad completa (incluye
la imagen original de la que salio cada recorte, en "source_image").

Uso:
    python scripts/convert_dataset.py
"""

import csv
import pathlib
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.core.preprocessing.cropping import crop_to_box  # noqa: E402
from src.utils.duplicates import build_duplicate_groups  # noqa: E402

RAW_DIR      = ROOT / "data" / "raw"
LABELED_DIR  = RAW_DIR / "labeled"
POOL_LOG     = RAW_DIR / "pool" / "pool_log.csv"
BBOX_LOG     = RAW_DIR / "bbox_log.csv"
OUTPUT       = ROOT / "data" / "processed"
LABELS_OUT   = OUTPUT / "dataset_labels.csv"
CROP_MARGIN  = 0.15

SPLITS       = ["train", "valid", "test"]
LABELS       = ["fall", "no_fall"]
SPLIT_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}
RANDOM_SEED  = 42


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


def load_bbox_rows() -> list[dict]:
    """Una fila por persona: filename, label, x1, y1, x2, y2, timestamp."""
    if not BBOX_LOG.exists():
        return []
    with open(BBOX_LOG, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_source_lookup() -> dict[str, str]:
    """filename -> procedencia real (pexels/video), segun pool_log.csv"""
    if not POOL_LOG.exists():
        return {}
    with open(POOL_LOG, encoding="utf-8") as f:
        return {row["filename"]: row["source"] for row in csv.DictReader(f)}


print("\n" + "=" * 60)
print("PASO 3: ARMADO DEL DATASET FINAL")
print("=" * 60)

rows = load_bbox_rows()

if not rows:
    print(f"\nERROR: {BBOX_LOG} esta vacio o no existe")
    print("  Etiqueta imagenes con label_tool.py (dibujando al menos una persona) antes de correr esto.")
    exit(1)

print(f"\nPersonas etiquetadas encontradas: {len(rows)}")
for label in LABELS:
    print(f"  {label}: {sum(1 for r in rows if r['label'] == label)}")

# Cada fila necesita: (1) procedencia conocida en pool_log.csv, y (2) que su
# imagen ya haya sido movida a data/raw/labeled/ (osea, se confirmo "Listo"
# para ella en label_tool.py). Si falta alguna, se corta en vez de adivinar.
source_lookup = load_source_lookup()
missing_source = sorted({r["filename"] for r in rows if r["filename"] not in source_lookup})
missing_labeled = sorted({r["filename"] for r in rows if not (LABELED_DIR / r["filename"]).exists()})

if missing_source:
    print("\nERROR: hay filas de bbox_log.csv sin procedencia conocida en pool_log.csv:")
    for name in missing_source:
        print(f"  {name}")
    print(
        "\nToda imagen debe entrar por scrape_dataset.py o extract_video_frames.py "
        "antes de etiquetarla con label_tool.py."
    )
    exit(1)

if missing_labeled:
    print(f"\nERROR: hay filas de bbox_log.csv cuya imagen no esta en {LABELED_DIR}:")
    for name in missing_labeled:
        print(f"  {name}")
    print(
        "\nTermina de etiquetar esas imagenes en label_tool.py y confirma "
        "'Listo, siguiente' para que se muevan a data/raw/labeled/."
    )
    exit(1)

# Numero de ocurrencia por filename (orden de aparicion en bbox_log.csv), para
# poder generar nombres de salida unicos cuando una misma imagen aporta mas
# de una persona (ej. foto_00abc12.jpg -> foto_00abc12_0.jpg, _1.jpg, ...).
occurrence: dict[str, int] = defaultdict(int)
for row in rows:
    row["_occurrence"] = occurrence[row["filename"]]
    occurrence[row["filename"]] += 1

# Agrupar imagenes casi-identicas (dHash) ANTES de dividir: un grupo entero
# va al mismo split, para que dos frames casi iguales no queden separados
# entre train y test (ver docstring del modulo).
print("\nAgrupando imagenes casi-duplicadas (dHash)...")
image_group = build_duplicate_groups(LABELED_DIR, {r["filename"] for r in rows})
for row in rows:
    row["_group"] = image_group[row["filename"]]

n_groups = len(set(image_group.values()))
n_grouped_images = sum(1 for g, count in Counter(image_group.values()).items() if count > 1)
print(
    f"  {len(image_group)} imagenes -> {n_groups} grupos "
    f"({len(image_group) - n_groups} imagenes fusionadas en un grupo con otra casi-identica)"
)

# Cada grupo se estratifica por su label dominante (el mas frecuente entre
# las personas de ese grupo) y su fuente (deberia ser uniforme dentro del
# grupo, ya que los casi-duplicados vienen del mismo video/foto original).
group_labels: dict[str, list[str]] = defaultdict(list)
group_sources: dict[str, list[str]] = defaultdict(list)
for row in rows:
    group_labels[row["_group"]].append(row["label"])
    group_sources[row["_group"]].append(source_lookup[row["filename"]])

group_dominant_label = {g: Counter(lbls).most_common(1)[0][0] for g, lbls in group_labels.items()}
group_source = {g: Counter(srcs).most_common(1)[0][0] for g, srcs in group_sources.items()}

# Limpiar y recrear estructura de salida
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)
for split in SPLITS:
    for label in LABELS:
        (OUTPUT / split / label).mkdir(parents=True, exist_ok=True)

label_rows: list[dict] = []
group_split: dict[str, str] = {}

for label in LABELS:
    label_groups = [g for g, lbl in group_dominant_label.items() if lbl == label]

    by_source: dict[str, list[str]] = defaultdict(list)
    for g in label_groups:
        by_source[group_source[g]].append(g)

    print(f"\n  {label}:")

    for source, source_groups in sorted(by_source.items()):
        source_splits = split_list(source_groups, SPLIT_RATIOS, RANDOM_SEED)
        for split, split_groups in source_splits.items():
            for g in split_groups:
                group_split[g] = split
        n_people = sum(len(group_labels[g]) for g in source_groups)
        print(
            f"    source={source}: {len(source_groups)} grupos / {n_people} personas "
            f"(train={len(source_splits['train'])}, "
            f"valid={len(source_splits['valid'])}, "
            f"test={len(source_splits['test'])} grupos)"
        )

for row in rows:
    row["_split"] = group_split[row["_group"]]

for label in LABELS:
    for split in SPLITS:
        split_items = [r for r in rows if r["label"] == label and r["_split"] == split]
        dest_dir = OUTPUT / split / label
        print(f"    [{split}/{label}] total: {len(split_items)}")

        for row in split_items:
            src_path = LABELED_DIR / row["filename"]
            box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))

            img = Image.open(src_path).convert("RGB")
            cropped = crop_to_box(img, box, margin=CROP_MARGIN)

            src_name = pathlib.Path(row["filename"])
            out_name = f"{src_name.stem}_{row['_occurrence']}{src_name.suffix}"
            dst = dest_dir / out_name
            cropped.save(dst)

            label_rows.append({
                "filename":     out_name,
                "source_image": row["filename"],
                "label":        label,
                "source":       source_lookup[row["filename"]],
                "split":        split,
                "timestamp":    datetime.now().isoformat(),
            })

with open(LABELS_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "source_image", "label", "source", "split", "timestamp"])
    writer.writeheader()
    writer.writerows(label_rows)

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
for split in SPLITS:
    fall    = len(list((OUTPUT / split / "fall").glob("*")))
    no_fall = len(list((OUTPUT / split / "no_fall").glob("*")))
    print(f"  [{split}] fall: {fall} | no_fall: {no_fall} | total: {fall + no_fall}")

# Desglose source x label x split, para verificar que la estratificacion
# por fuente no dejo un split sin representacion de alguna de las dos.
print("\nDesglose por fuente (source x label x split):")
sources = sorted({row["source"] for row in label_rows})
for source in sources:
    for label in LABELS:
        counts = {
            split: sum(
                1 for row in label_rows
                if row["source"] == source and row["label"] == label and row["split"] == split
            )
            for split in SPLITS
        }
        print(f"  {source}/{label}: " + " | ".join(f"{s}={counts[s]}" for s in SPLITS))

print(f"\nDataset listo en      : {OUTPUT}")
print(f"Trazabilidad en       : {LABELS_OUT}")
print(f"Total de ejemplos     : {len(label_rows)}")
