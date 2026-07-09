"""
scripts/find_inconsistent_duplicates.py

Detecta grupos de imagenes casi-duplicadas (mismo dHash que usa
convert_dataset.py para evitar fuga entre splits) donde las personas
etiquetadas tienen labels DISTINTOS (fall y no_fall a la vez dentro del
mismo grupo) — señal de un posible error de etiquetado, ya que frames casi
identicos deberian etiquetarse igual.

No modifica nada: solo reporta, y guarda una imagen de comparacion lado a
lado por cada grupo conflictivo en data/raw/duplicate_review/ para revision
visual rapida. Para corregir un caso, volve a etiquetar esa imagen puntual
con label_tool.py (o edita/borra la fila correspondiente en bbox_log.csv).

Uso:
    python scripts/find_inconsistent_duplicates.py
"""

import csv
import pathlib
import sys
from collections import defaultdict

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.utils.duplicates import build_duplicate_groups  # noqa: E402

RAW_DIR     = ROOT / "data" / "raw"
LABELED_DIR = RAW_DIR / "labeled"
BBOX_LOG    = RAW_DIR / "bbox_log.csv"
REVIEW_DIR  = RAW_DIR / "duplicate_review"


def load_bbox_rows() -> list[dict]:
    if not BBOX_LOG.exists():
        return []
    with open(BBOX_LOG, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_side_by_side(filenames: list[str], dst: pathlib.Path, height: int = 300) -> None:
    imgs = [Image.open(LABELED_DIR / name).convert("RGB") for name in filenames[:4]]
    resized = [img.resize((int(img.width * height / img.height), height)) for img in imgs]
    total_width = sum(im.width for im in resized) + 10 * (len(resized) - 1)

    canvas = Image.new("RGB", (total_width, height), "white")
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width + 10
    canvas.save(dst, quality=90)


def main() -> None:
    rows = load_bbox_rows()
    if not rows:
        print(f"{BBOX_LOG} esta vacio o no existe.")
        return

    filenames = {r["filename"] for r in rows if (LABELED_DIR / r["filename"]).exists()}
    missing = {r["filename"] for r in rows} - filenames
    if missing:
        print(f"Aviso: {len(missing)} filas de bbox_log.csv no tienen imagen en {LABELED_DIR}, se ignoran.")

    print(f"Agrupando {len(filenames)} imagenes por casi-duplicado (dHash)...")
    groups = build_duplicate_groups(LABELED_DIR, filenames)

    group_files: dict[str, set] = defaultdict(set)
    group_labels: dict[str, set] = defaultdict(set)
    labels_by_file: dict[str, list] = defaultdict(list)

    for r in rows:
        if r["filename"] not in groups:
            continue
        g = groups[r["filename"]]
        group_files[g].add(r["filename"])
        group_labels[g].add(r["label"])
        labels_by_file[r["filename"]].append(r["label"])

    multi_image_groups = {g: f for g, f in group_files.items() if len(f) > 1}
    conflicting = {g: f for g, f in multi_image_groups.items() if len(group_labels[g]) > 1}

    print(f"\nGrupos casi-duplicados con mas de una imagen: {len(multi_image_groups)}")
    print(f"De esos, con labels EN CONFLICTO (fall y no_fall a la vez): {len(conflicting)}")

    if not conflicting:
        print("\nNo se encontraron conflictos. Nada para revisar.")
        return

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for old in REVIEW_DIR.glob("conflicto_*.jpg"):
        old.unlink()

    print(f"\nDetalle (comparaciones guardadas en {REVIEW_DIR}):\n")
    for i, (g, files) in enumerate(sorted(conflicting.items()), start=1):
        files = sorted(files)
        print(f"  [{i}] grupo con {len(files)} imagenes:")
        for fname in files:
            print(f"        {fname}: labels = {labels_by_file[fname]}")

        out_path = REVIEW_DIR / f"conflicto_{i:02d}.jpg"
        save_side_by_side(files, out_path)
        print(f"        -> {out_path.name}")

    print(f"\n{len(conflicting)} comparaciones guardadas en {REVIEW_DIR}")
    print("Revisalas y, si corresponde, volve a etiquetar esas imagenes con label_tool.py")
    print("(o corregi/borra la fila correspondiente directamente en bbox_log.csv).")


if __name__ == "__main__":
    main()
