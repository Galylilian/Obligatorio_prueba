"""
scripts/scrape_dataset.py

Scraper de imágenes para el dataset de detección de caídas.
Usa 2 fuentes con filtro de contenido seguro:
    - Pexels  (API key requerida)
    - Pixabay (API key requerida)

Objetivo: 360 imágenes totales → 180 por clase
12 queries por clase × 8 imágenes × 2 fuentes = ~180/clase

Estrategia de etiquetado: Weak Supervision
La etiqueta se asigna por heurística basada en los términos de búsqueda.
La deduplicación es cross-fuente por hash MD5 — ninguna imagen se repite.
Cada imagen queda registrada en data/raw/scraped/scraping_log.csv.

Requisitos:
    pip install requests pillow python-dotenv

API Keys (ambas gratuitas):
    Pexels  → https://www.pexels.com/api/
    Pixabay → https://pixabay.com/api/docs/

Agregar al .env:
    PEXELS_API_KEY=...
    PIXABAY_API_KEY=...

Uso:
    python scripts/scrape_dataset.py
"""

import os
import csv
import time
import uuid
import hashlib
import shutil
import requests
import pathlib
from datetime import datetime
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# =============================
# PATHS
# =============================
ROOT       = pathlib.Path(__file__).resolve().parents[1]
SCRAPED_DIR = ROOT / "data" / "raw" / "scraped"
LOG_FILE   = SCRAPED_DIR / "scraping_log.csv"

# =============================
# CARGAR API KEYS
# =============================
load_dotenv(ROOT / ".env")

PEXELS_API_KEY  = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

# Validar que al menos una key está disponible
active_sources = []
if PEXELS_API_KEY:
    active_sources.append("pexels")
if PIXABAY_API_KEY:
    active_sources.append("pixabay")

if not active_sources:
    raise ValueError(
        "❌ No hay ninguna API key configurada en el .env\n"
        "   Necesitás al menos una de:\n"
        "   PEXELS_API_KEY, PIXABAY_API_KEY"
    )

# =============================
# QUERIES POR CLASE Y FUENTE
#
# Criterio de selección:
#
# FALL:
#   Escenarios reales y variados: interior, exterior, adultos mayores,
#   accidentes domésticos, escaleras, emergencias médicas.
#   Se evitan términos deportivos o artísticos que traen
#   imágenes fuera del dominio (skateboard fall, dance fall).
#
# NO_FALL:
#   Distintas posturas para que el modelo aprenda que "no caída"
#   no es solo "persona de pie": sentada, caminando, con bastón.
#   Se incluyen adultos mayores para balancear con los queries de fall.
# =============================

QUERIES = {
    "fall": [
        "elderly person fallen floor home",
        "person lying floor emergency",
        "old person collapsed ground indoors",
        "person fallen bathroom floor",
        "senior citizen fall accident home",
        "person fallen sidewalk street",
        "man collapsed pavement outside",
        "person lying ground outdoors accident",
        "person fallen stairs",
        "accident fall staircase",
        "paramedic helping person fallen floor",
        "emergency fall elderly ground",
    ],
    "no_fall": [
        "elderly person standing home",
        "senior woman standing kitchen",
        "old man standing living room",
        "elderly person walking indoors",
        "senior person walking corridor",
        "person walking hallway",
        "elderly person sitting chair",
        "senior person sitting sofa",
        "elderly person walking cane indoors",
        "senior person walking walker",
        "elderly person standing outside",
        "senior couple walking park",
    ],
}

# =============================
# CONFIGURACIÓN
# =============================

# Objetivo: 360 imágenes totales → 180 por clase
# 12 queries por clase → ~8 imágenes por query (entre las 2 fuentes)
# Se pide 8 por query por fuente → puede haber duplicados entre fuentes
IMAGES_PER_QUERY   = 8     # imágenes por query por fuente
PAGES_PER_QUERY    = 1     # una sola página por query
MIN_WIDTH          = 224
MIN_HEIGHT         = 224
DOWNLOAD_TIMEOUT   = 10
SLEEP_BETWEEN_IMAGES  = 0.5
SLEEP_BETWEEN_QUERIES = 3.0


# =============================
# UTILIDADES COMUNES
# =============================

def is_valid_image(img: Image.Image) -> bool:
    """Filtra imágenes demasiado pequeñas o con ratio extremo."""
    if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
        return False
    ratio = img.width / img.height
    if ratio > 4.0 or ratio < 0.25:
        return False
    return True


def image_hash(content: bytes) -> str:
    """MD5 para deduplicación cross-fuente y cross-query."""
    return hashlib.md5(content).hexdigest()


def download_image(url: str) -> bytes | None:
    """Descarga imagen con timeout y validación de content-type."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, headers=headers)
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            if "image" in content_type:
                return response.content
    except Exception:
        pass
    return None


def process_and_save(
    content: bytes,
    label: str,
    source: str,
    query: str,
    page: int,
    extra_meta: dict,
    dest_dir: pathlib.Path,
    seen_hashes: set,
    log_rows: list,
    counters: dict,
) -> bool:
    """
    Deduplica, valida y guarda una imagen.
    Retorna True si fue guardada, False si fue descartada.
    Centraliza la lógica para que las 3 fuentes usen el mismo pipeline.
    """
    h = image_hash(content)
    if h in seen_hashes:
        counters["duplicates"] += 1
        return False
    seen_hashes.add(h)

    try:
        img = Image.open(BytesIO(content)).convert("RGB")
        if not is_valid_image(img):
            counters["invalid"] += 1
            return False
    except Exception:
        counters["invalid"] += 1
        return False

    filename = f"{label}_{uuid.uuid4().hex[:8]}.jpg"
    dest_path = dest_dir / filename

    try:
        img.save(dest_path, "JPEG", quality=90)
    except Exception:
        return False

    log_rows.append({
        "filename":  filename,
        "label":     label,
        "source":    source,
        "query":     query,
        "page":      page,
        "hash":      h,
        "timestamp": datetime.now().isoformat(),
        **extra_meta,
    })

    return True


# =============================
# PEXELS
# =============================

def search_pexels(query: str, per_page: int, page: int) -> list:
    url     = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params  = {
        "query":       query,
        "per_page":    per_page,
        "page":        page,
        "orientation": "portrait",
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get("photos", [])
        elif r.status_code == 429:
            print("  ⏳ Pexels rate limit — esperando 60s")
            time.sleep(60)
            return search_pexels(query, per_page, page)
        else:
            print(f"  ❌ Pexels error: {r.status_code}")
            return []
    except Exception as e:
        print(f"  ❌ Pexels request error: {e}")
        return []


def scrape_pexels(label, query, page, dest_dir, seen_hashes, log_rows, counters):
    photos = search_pexels(query, IMAGES_PER_QUERY, page)
    downloaded = 0
    for photo in photos:
        src  = photo.get("src", {})
        url  = src.get("large2x") or src.get("large") or src.get("medium")
        if not url:
            continue
        content = download_image(url)
        if not content:
            counters["failed"] += 1
            continue
        saved = process_and_save(
            content, label, "pexels", query, page,
            {"source_id": str(photo.get("id", "")), "url": url},
            dest_dir, seen_hashes, log_rows, counters,
        )
        if saved:
            downloaded += 1
            counters["total"][label] += 1
        time.sleep(SLEEP_BETWEEN_IMAGES)
    return downloaded


# =============================
# PIXABAY
# =============================

def search_pixabay(query: str, per_page: int, page: int) -> list:
    url    = "https://pixabay.com/api/"
    params = {
        "key":        PIXABAY_API_KEY,
        "q":          query,
        "image_type": "photo",
        "safesearch": "true",       # filtro de contenido activado
        "per_page":   min(per_page, 200),
        "page":       page,
        "lang":       "en",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get("hits", [])
        elif r.status_code == 429:
            print("  ⏳ Pixabay rate limit — esperando 60s")
            time.sleep(60)
            return search_pixabay(query, per_page, page)
        else:
            print(f"  ❌ Pixabay error: {r.status_code}")
            return []
    except Exception as e:
        print(f"  ❌ Pixabay request error: {e}")
        return []


def scrape_pixabay(label, query, page, dest_dir, seen_hashes, log_rows, counters):
    hits = search_pixabay(query, IMAGES_PER_QUERY, page)
    downloaded = 0
    for hit in hits:
        # largeImageURL > webformatURL como fallback
        url = hit.get("largeImageURL") or hit.get("webformatURL")
        if not url:
            continue
        content = download_image(url)
        if not content:
            counters["failed"] += 1
            continue
        saved = process_and_save(
            content, label, "pixabay", query, page,
            {"source_id": str(hit.get("id", "")), "url": url},
            dest_dir, seen_hashes, log_rows, counters,
        )
        if saved:
            downloaded += 1
            counters["total"][label] += 1
        time.sleep(SLEEP_BETWEEN_IMAGES)
    return downloaded



# =============================
# MAPA DE FUENTES ACTIVAS
# Permite agregar o quitar fuentes fácilmente
# =============================
SCRAPERS = {
    "pexels":  scrape_pexels,
    "pixabay": scrape_pixabay,
}


# =============================
# SCRAPER PRINCIPAL
# =============================

def scrape_images():

    # =============================
    # LIMPIAR CARPETAS ANTERIORES
    # =============================
    print("\n🧹 Limpiando carpetas anteriores...")
    for label in QUERIES:
        label_dir = SCRAPED_DIR / label
        if label_dir.exists():
            shutil.rmtree(label_dir)
            print(f"  🗑️  {label_dir} eliminada")
        label_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {label_dir} creada")

    log_rows   = []
    seen_hashes = set()   # deduplicación GLOBAL cross-fuente y cross-clase
    counters   = {
        "total":      {label: 0 for label in QUERIES},
        "duplicates": 0,
        "invalid":    0,
        "failed":     0,
    }

    print("\n" + "=" * 60)
    print("SCRAPER DE IMÁGENES — Fall Detector Dataset")
    print(f"Fuentes activas : {', '.join(active_sources)}")
    print(f"Queries/clase   : {len(next(iter(QUERIES.values())))}")
    print(f"Imágenes/query  : {IMAGES_PER_QUERY} x {PAGES_PER_QUERY} páginas")
    print("Etiquetado      : Weak Supervision")
    print("=" * 60)

    for label, queries in QUERIES.items():

        print(f"\n{'=' * 40}")
        print(f"📁 Clase: {label.upper()}")
        print(f"{'=' * 40}")

        dest_dir = SCRAPED_DIR / label

        for query in queries:

            print(f"\n  🔍 '{query}'")

            for source in active_sources:

                scraper = SCRAPERS[source]
                source_downloaded = 0

                for page in range(1, PAGES_PER_QUERY + 1):
                    n = scraper(
                        label, query, page,
                        dest_dir, seen_hashes, log_rows, counters
                    )
                    source_downloaded += n
                    if n == 0:
                        break   # sin más resultados en esta fuente

                print(f"    [{source}] ✅ {source_downloaded} imágenes")

            time.sleep(SLEEP_BETWEEN_QUERIES)

    # =============================
    # GUARDAR CSV DE LOG
    # =============================
    SCRAPED_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "filename", "label", "source", "query",
            "page", "source_id", "url", "hash", "timestamp"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    # =============================
    # RESUMEN FINAL
    # =============================
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    for label, count in counters["total"].items():
        print(f"  {label:<10} : {count} imágenes")
    print(f"  duplicadas : {counters['duplicates']}")
    print(f"  inválidas  : {counters['invalid']}")
    print(f"  fallidas   : {counters['failed']}")
    print(f"  log        : {LOG_FILE}")
    print("=" * 60)
    print("\n✅ Corré convert_dataset.py para dividir en train/valid/test")


if __name__ == "__main__":
    scrape_images()