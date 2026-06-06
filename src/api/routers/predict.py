from fastapi import APIRouter, UploadFile
from PIL import Image
import torch

from src.core.inference import get_eval_transform, get_inference_model, get_label_encoder
from src.utils.label_encoder import decode_prediction
from src.utils.logger import get_logger

logger = get_logger("predict")

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile):
    logger.info(f"Request recibido: {file.filename}")

    try:
        model = get_inference_model()
        label_encoder = get_label_encoder()
        transform = get_eval_transform()

        image = Image.open(file.file).convert("RGB")
        image = transform(image).unsqueeze(0).to(next(model.parameters()).device)

        logger.info("Imagen procesada correctamente")

        with torch.no_grad():
            pred = model(image).argmax().item()

        logger.info(f"Predicción realizada: {pred}")

        return {
            "prediction": pred,
            "label": decode_prediction(label_encoder, pred),
        }

    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")

        return {
            "error": "Error procesando la imagen"
        }
