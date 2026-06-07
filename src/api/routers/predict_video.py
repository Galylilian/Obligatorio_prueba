from pathlib import Path
import shutil
import json

from fastapi import APIRouter, UploadFile, File

from src.utils.video_detection import detect_falls_from_video
from src.utils.logger import get_logger


router = APIRouter()
logger = get_logger("predict_video")

UPLOAD_DIR = Path("data/video/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    logger.info(f"Video recibido: {file.filename}")

    try:
        # =============================
        # GUARDAR VIDEO
        # =============================
        video_path = UPLOAD_DIR / file.filename

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # =============================
        # PROCESAR VIDEO
        # =============================
        output_path = UPLOAD_DIR / f"{video_path.stem}_results.json"

        detect_falls_from_video(
            video_path=str(video_path),
            output_path=str(output_path),
            save_frames=True,
        )

        # =============================
        # LEER RESULTADOS
        # =============================
        with open(output_path) as f:
            results = json.load(f)

        # Resumen
        total = len(results)
        falls = [r for r in results if r["is_fall"]]

        return {
            "message": "Video procesado correctamente",
            "total_frames": total,
            "falls_detected": len(falls),
            "preview": results[:20],
        }

    except Exception as e:
        logger.error(f"Error procesando video: {str(e)}")

        return {
            "error": str(e)
        }

