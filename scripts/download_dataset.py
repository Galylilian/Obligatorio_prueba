import shutil
from pathlib import Path
from dotenv import load_dotenv
import os
from roboflow import Roboflow

# =============================
# CONFIG
# =============================
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("⚠️ Falta ROBOFLOW_API_KEY en .env")

rf = Roboflow(api_key=api_key)

# =============================
# PATHS
# =============================
RAW_DIR = ROOT / "data" / "raw"

# limpiar carpeta raw
if RAW_DIR.exists():
    shutil.rmtree(RAW_DIR)

RAW_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 70)
print("DESCARGA DE DATASET (VERSION ESTABLE)")
print("=" * 70)

# =============================
# DESCARGA SOLO DS1 ✅
# =============================
print("\n⬇️ Descargando fall-detection-raskl")

try:
    dataset = (
        rf.workspace("fall-detection")
        .project("fall-detection-raskl")
        .version(1)
        .download("folder")  # ✅ formato correcto
    )

    dest = RAW_DIR / "ds1"
    shutil.copytree(Path(dataset.location), dest)

    print(f"✅ Dataset guardado en: {dest}")

except Exception as e:
    print(f"❌ Error al descargar dataset: {e}")
    print("👉 Descargalo manualmente desde Roboflow y colócalo en data/raw/ds1")

print("\n✅ Proceso finalizado: dataset listo para usar")