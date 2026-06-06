from fastapi import APIRouter, UploadFile
from PIL import Image
import torch
from torchvision import transforms

from src.core.model import get_model
from src.settings.config import MODEL_PATH, DEVICE
from src.utils.logger import get_logger

# === logger ===
logger = get_logger("predict")

router = APIRouter()

# === cargar modelo ===
logger.info(f"Cargando modelo desde: {MODEL_PATH}")

model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

logger.info("Modelo cargado correctamente ✅")

# === transformaciones ===

from src.core.preprocessing.transforms import get_test_transforms

transform = get_test_transforms()



@router.post("/predict")
async def predict(file: UploadFile):
    logger.info(f"Request recibido: {file.filename}")

    try:
        # cargar imagen
        image = Image.open(file.file).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)

        logger.info("Imagen procesada correctamente")

        # inferencia
        with torch.no_grad():
            pred = model(image).argmax().item()

        logger.info(f"Predicción realizada: {pred}")

        return {"prediction": pred}

    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")

        return {
            "error": "Error procesando la imagen"
        }
