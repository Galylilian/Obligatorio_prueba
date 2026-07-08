"""
scripts/extract_video_frames.py

PASO 1b: Extrae frames crudos de videos propios hacia el pool sin etiqueta.

Lee todos los videos de data/video/input/ y guarda cada frame como imagen
en data/raw/pool/, junto con Pexels, listas para etiquetar con label_tool.py.

No usa el modelo para preseleccionar frames: se extraen todos, para que el
etiquetado manual sea el que decida fall/no_fall (evita el sesgo de que el
modelo ya entrenado filtre qué le mostramos al etiquetador).

Uso:
    python scripts/extract_video_frames.py
    python scripts/extract_video_frames.py --interval 5   (1 frame cada 5 segundos)
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from datetime import datetime

import cv2

ROOT      = pathlib.Path(__file__).resolve().parents[1]
POOL_DIR  = ROOT / "data" / "raw" / "pool"
LOG_FILE  = POOL_DIR / "pool_log.csv"
VIDEO_DIR = ROOT / "data" / "video" / "input"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

FIELDNAMES = [
    "filename", "source", "timestamp",
    "search_query", "source_id", "url", "hash",
    "video_file", "frame_idx", "time_sec",
]


def load_existing_log() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_log(log_rows: list[dict]) -> None:
    tmp_file = LOG_FILE.with_suffix(".csv.tmp")
    with open(tmp_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(log_rows)
    tmp_file.replace(LOG_FILE)


def list_videos() -> list[pathlib.Path]:
    if not VIDEO_DIR.exists():
        return []
    return sorted(
        f for f in VIDEO_DIR.iterdir()
        if f.suffix.lower() in VIDEO_EXTS
    )


def extract_frames(video_path: pathlib.Path, interval: float, log_rows: list[dict]) -> int:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    every = max(1, round(fps * interval))
    stem = video_path.stem
    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every == 0:
            filename = f"vid_{stem}_{frame_idx:06d}.jpg"
            cv2.imwrite(str(POOL_DIR / filename), frame)

            log_rows.append({
                "filename":     filename,
                "source":       "video",
                "timestamp":    datetime.now().isoformat(),
                "search_query": "",
                "source_id":    "",
                "url":          "",
                "hash":         "",
                "video_file":   video_path.name,
                "frame_idx":    str(frame_idx),
                "time_sec":     str(round(frame_idx / fps, 2)),
            })
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae frames de video al pool sin etiqueta")
    parser.add_argument("--interval", type=float, default=10, help="Guardar 1 frame cada N segundos (default: 10)")
    args = parser.parse_args()

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    videos = list_videos()

    if not videos:
        print(f"No hay videos en {VIDEO_DIR}")
        return

    log_rows = load_existing_log()
    total = 0

    print(f"\nExtrayendo frames | {len(videos)} video(s) | cada {args.interval}s")
    print("=" * 60)

    for video in videos:
        saved = extract_frames(video, args.interval, log_rows)
        print(f"  {video.name}: {saved} frames")
        total += saved

    write_log(log_rows)

    print("=" * 60)
    print(f"Total frames guardados : {total}")
    print(f"Pool total (log)       : {len(log_rows)}")
    print(f"Directorio             : {POOL_DIR}")
    print("\nPASO 1b completo.")
    print("Siguiente paso:")
    print("  python scripts/label_tool.py")


if __name__ == "__main__":
    main()
