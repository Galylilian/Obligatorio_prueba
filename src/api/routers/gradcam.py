from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse

from PIL import Image
import torch
import numpy as np
import cv2
import io

from src.core.inference import get_eval_transform, get_inference_model
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger("gradcam")

_grad_cam = None


def _get_gradcam():
    global _grad_cam
    if _grad_cam is None:
        model = get_inference_model()
        target_layer = model.layer4[-1].conv2
        _grad_cam = GradCAM(model, target_layer)
    return _grad_cam


@router.post("/gradcam")
async def generate_gradcam(file: UploadFile):
    logger.info(f"Request Grad-CAM recibido: {file.filename}")

    try:
        model = get_inference_model()
        grad_cam = _get_gradcam()
        transform = get_eval_transform()

        image = Image.open(file.file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

        logger.info("Imagen procesada correctamente")

        with torch.no_grad():
            output = model(input_tensor)
            pred_class = output.argmax().item()

        logger.info(f"Clase predicha: {pred_class}")

        cam = grad_cam.generate(input_tensor)

        logger.info("Grad-CAM generado")

        img_np = np.array(image.resize((224, 224)))
        result = overlay_heatmap(img_np, cam)

        _, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        )

        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )

    except Exception as e:
        logger.error(f"Error en Grad-CAM: {str(e)}")

        return {
            "error": "Error generando Grad-CAM"
        }
