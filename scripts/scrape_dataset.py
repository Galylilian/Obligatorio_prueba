"""
scripts/scrape_dataset.py

PASO 1: Descarga imagenes sin etiqueta desde Pexels al pool.

Las imagenes quedan en data/raw/pool/ listas para etiquetar.
La deduplicacion es por hash MD5.

Requisitos:
    pip install requests pillow python-dotenv tqdm

Agregar al .env:
    PEXELS_API_KEY=...

Uso:
    python scripts/scrape_dataset.py
"""

from __future__ import annotations

import csv
import hashlib
import os
import pathlib
import time
import uuid
from datetime import datetime
from io import BytesIO

import requests
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

ROOT         = pathlib.Path(__file__).resolve().parents[1]
POOL_DIR     = ROOT / "data" / "raw" / "pool"
LOG_FILE     = POOL_DIR / "pool_log.csv"
QUERIES_FILE = ROOT / "queries" / "people.txt"

load_dotenv(ROOT / ".env")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

if not PEXELS_API_KEY:
    raise ValueError(
        "No hay PEXELS_API_KEY en el .env\n"
        "Conseguila gratis en https://www.pexels.com/api/"
    )

FIELDNAMES = [
    "filename", "source", "timestamp",
    "search_query", "source_id", "url", "hash",
    "video_file", "frame_idx", "time_sec",
]

IMAGES_PER_QUERY = 20
MIN_WIDTH        = 224
MIN_HEIGHT       = 224
DOWNLOAD_TIMEOUT = 10
SLEEP_BETWEEN    = 0.3


def load_queries() -> list[str]:
    if not QUERIES_FILE.exists():
        raise FileNotFoundError(f"No se encontro {QUERIES_FILE}")
    queries = [
        line.strip()
        for line in QUERIES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not queries:
        raise ValueError("queries/people.txt no tiene queries validas")
    return queries


def image_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def is_valid(img: Image.Image) -> bool:
    if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
        return False
    ratio = img.width / img.height
    return 0.25 <= ratio <= 4.0


def search_pexels(query: str, per_page: int) -> list:
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": per_page, "orientation": "portrait"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get("photos", [])
        if r.status_code == 429:
            print("  Pexels rate limit - esperando 60s")
            time.sleep(60)
            return search_pexels(query, per_page)
        print(f"  Pexels error: {r.status_code}")
    except Exception as e:
        print(f"  Error de red: {e}")
    return []


def download_image(url: str) -> bytes | None:
    try:
        r = requests.get(
            url, timeout=DOWNLOAD_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
            return r.content
    except Exception:
        pass
    return None


def load_existing_log() -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    hashes: set[str] = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                hashes.add(row.get("hash", ""))
    return rows, hashes


def write_log(log_rows: list[dict]) -> None:
    tmp_file = LOG_FILE.with_suffix(".csv.tmp")
    with open(tmp_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(log_rows)
    tmp_file.replace(LOG_FILE)


def scrape_pool() -> None:
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    queries = load_queries()

    log_rows, seen_hashes = load_existing_log()
    counters = {"saved": 0, "duplicates": 0, "invalid": 0, "failed": 0}

    existing = len(log_rows)
    if existing:
        print(f"Pool existente: {existing} imagenes (continuando)")

    print(f"\nDescargando | Pexels | {len(queries)} queries | {IMAGES_PER_QUERY} por query")
    print("=" * 60)

    for query in queries:
        photos = search_pexels(query, IMAGES_PER_QUERY)

        for photo in tqdm(photos, desc=query[:38], unit="img", leave=False):
            src = photo.get("src", {})
            url = src.get("large2x") or src.get("large") or src.get("medium")
            if not url:
                continue

            content = download_image(url)
            if not content:
                counters["failed"] += 1
                continue

            h = image_hash(content)
            if h in seen_hashes:
                counters["duplicates"] += 1
                continue
            seen_hashes.add(h)

            try:
                img = Image.open(BytesIO(content)).convert("RGB")
                if not is_valid(img):
                    counters["invalid"] += 1
                    continue
            except Exception:
                counters["invalid"] += 1
                continue

            filename = f"img_{uuid.uuid4().hex[:10]}.jpg"
            img.save(POOL_DIR / filename, "JPEG", quality=90)

            log_rows.append({
                "filename":     filename,
                "source":       "pexels",
                "timestamp":    datetime.now().isoformat(),
                "search_query": query,
                "source_id":    str(photo.get("id", "")),
                "url":          url,
                "hash":         h,
                "video_file":   "",
                "frame_idx":    "",
                "time_sec":     "",
            })
            counters["saved"] += 1
            time.sleep(SLEEP_BETWEEN)

    write_log(log_rows)

    print("\n" + "=" * 60)
    print("RESUMEN")
    print(f"  guardadas   : {counters['saved']}")
    print(f"  duplicadas  : {counters['duplicates']}")
    print(f"  invalidas   : {counters['invalid']}")
    print(f"  fallidas    : {counters['failed']}")
    print(f"  pool total  : {len(log_rows)}")
    print(f"  directorio  : {POOL_DIR}")
    print("=" * 60)
    print("\nPASO 1 completo.")
    print("Siguiente paso:")
    print("  python scripts/extract_video_frames.py   (opcional: sumar frames de video al pool)")
    print("  python scripts/label_tool.py")
    print("  python scripts/convert_dataset.py")


if __name__ == "__main__":
    scrape_pool()
