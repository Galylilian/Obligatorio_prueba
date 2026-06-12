from fastapi import APIRouter, UploadFile, Depends
from sqlalchemy.orm import Session
from PIL import Image

from src.core.classification import ImageClassifier
from src.utils.logger import get_logger
from src.db.database import get_db
from src.db.models import Prediction
from src.settings.config import MODEL_TYPE

# === logger ===
logger = get_logger("predict")

router = APIRouter()

# Instancia global (se carga una sola vez)
classifier = ImageClassifier()

logger.info("Modelo cargado correctamente ✅")


@router.post("/predict")
async def predict(file: UploadFile, db: Session = Depends(get_db)):
    logger.info(f"Request recibido: {file.filename}")

    try:
        # Cargar imagen
        image = Image.open(file.file).convert("RGB")

        logger.info("Imagen cargada correctamente")

        # Ejecutar predicción
        prediction = classifier.predict([image])["images"][0]

        logger.info(f"Predicción: {prediction}")

        # =============================
        # GUARDAR EN BASE DE DATOS ✅
        # =============================
        record = Prediction(
            filename=file.filename,
            label=prediction["label"],
            confidence=prediction["confidence"],
            model_type=MODEL_TYPE,
        )
        db.add(record)
        db.commit()

        return {
            "label": prediction["label"],
            "confidence": prediction["confidence"],
        }

    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")

        return {
            "error": "Error procesando la imagen"
        }