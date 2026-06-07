from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image

from src.analytics.repository import record_prediction
from src.core.inference import (
    get_eval_transform,
    get_inference_model,
    get_label_encoder,
    predict_with_confidence,
)
from src.utils.logger import get_logger

logger = get_logger("predict")

router = APIRouter()


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    person_id: str | None = Form(None),
    source: str | None = Form(None),
):
    logger.info(f"Request recibido: {file.filename} person_id={person_id}")

    try:
        model = get_inference_model()
        label_encoder = get_label_encoder()
        transform = get_eval_transform()

        image = Image.open(file.file).convert("RGB")
        image = transform(image).unsqueeze(0).to(next(model.parameters()).device)

        logger.info("Imagen procesada correctamente")

        pred_idx, label, confidence, probabilities = predict_with_confidence(
            model, image, label_encoder
        )

        logger.info(f"Prediccion realizada: {pred_idx} ({label}, conf={confidence:.3f})")

        try:
            record_prediction(
                label=label,
                confidence=confidence,
                person_id=person_id,
                source=source or file.filename,
            )
        except Exception as exc:
            logger.warning(f"No se pudo registrar evento analitico: {exc}")

        return {
            "prediction": pred_idx,
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "person_id": person_id,
        }

    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")

        return {
            "error": "Error procesando la imagen"
        }
