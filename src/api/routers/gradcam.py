from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse

from PIL import Image
import torch
import numpy as np
import cv2
import io

from src.core.model import get_model
from src.core.gradcam import GradCAM, overlay_heatmap
from src.core.preprocessing.transforms import get_test_transforms
from src.settings.config import MODEL_PATH, DEVICE
from src.utils.logger import get_logger

# =============================
# ROUTER Y LOGGER
# =============================
router = APIRouter()
logger = get_logger("gradcam")

# =============================
# CARGAR MODELO (UNA VEZ)
# =============================
logger.info(f"Cargando modelo para Grad-CAM desde: {MODEL_PATH}")

model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

logger.info("Modelo cargado correctamente ✅")

# =============================
# CONFIGURAR GRADCAM
# =============================
target_layer = model.layer4[-1]
grad_cam = GradCAM(model, target_layer)

# =============================
# TRANSFORM (SIN SKEW ✅)
# =============================
transform = get_test_transforms()

# =============================
# ENDPOINT
# =============================
@router.post("/gradcam")
async def generate_gradcam(file: UploadFile):

    logger.info(f"Request Grad-CAM recibido: {file.filename}")

    try:
        # =========================
        # CARGAR IMAGEN
        # =========================
        image = Image.open(file.file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        logger.info("Imagen procesada correctamente")

        # =========================
        # PREDICCIÓN (sin gradientes)
        # =========================
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = output.argmax().item()

        logger.info(f"Clase predicha: {pred_class}")

        # =========================
        # GENERAR GRADCAM (con gradientes)
        # =========================
        cam = grad_cam.generate(input_tensor)

        logger.info("Grad-CAM generado")

        # =========================
        # NORMALIZACIÓN SEGURA
        # =========================
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()

        # =========================
        # OVERLAY
        # =========================
        img_np = np.array(image.resize((224, 224)))
        result = overlay_heatmap(img_np, cam)

        # =========================
        # CONVERTIR A BYTES
        # =========================
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