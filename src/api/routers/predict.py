from fastapi import APIRouter, UploadFile
from PIL import Image

from src.core.classification import ImageClassifier
from src.utils.logger import get_logger

# === logger ===
logger = get_logger("predict")

router = APIRouter()

# Instancia global (se carga una sola vez)
classifier = ImageClassifier()

logger.info("Modelo cargado correctamente ✅")


@router.post("/predict")
async def predict(file: UploadFile):
    logger.info(f"Request recibido: {file.filename}")

    try:
        # Cargar imagen
        image = Image.open(file.file).convert("RGB")

        logger.info("Imagen cargada correctamente")

        # Ejecutar predicción
        prediction = classifier.predict([image])["images"][0]

        logger.info(f"Predicción: {prediction}")

        return {
            "label": prediction["label"],
            "confidence": prediction["confidence"],
        }

    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")

        return {
            "error": "Error procesando la imagen"
        }
