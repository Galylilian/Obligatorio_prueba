from fastapi import APIRouter
import json
from pathlib import Path

router = APIRouter()

# ✅ ruta absoluta para evitar problemas en Docker
BASE_DIR = Path(__file__).resolve().parents[3]
METRICS_FILE = BASE_DIR / "metrics.json"


@router.get("/dashboard/stats")
def get_stats():

    # =============================
    # VALIDAR EXISTENCIA
    # =============================
    if not METRICS_FILE.exists():
        return {
            "error": "No hay métricas disponibles. Ejecuta evaluate.py primero.",
            "model": {"splits": {"valid": {}}}
        }

    # =============================
    # LEER MÉTRICAS REALES
    # =============================
    try:
        with open(METRICS_FILE) as f:
            metrics = json.load(f)

    except Exception as e:
        return {
            "error": f"No se pudieron leer las métricas: {str(e)}",
            "model": {"splits": {"valid": {}}}
        }

    # =============================
    # RESPUESTA PARA STREAMLIT ✅
    # =============================
    return {
        # 🔹 estos pueden quedar en 0 (no vienen del modelo)
        "falls_today": 0,
        "falls_week": 0,
        "high_risk_persons": 0,

        "analytics_enabled": True,

        # ✅ estructura EXACTA que espera tu Streamlit
        "model": {
            "splits": {
                "valid": metrics   # 🔥 tu metrics.json real acá
            }
        }
    }

