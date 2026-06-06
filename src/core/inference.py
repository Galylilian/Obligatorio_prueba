"""Carga lazy del modelo y artefactos para inferencia (API)."""

from __future__ import annotations

from functools import lru_cache

import torch

from src.core.model import get_model
from src.preprocessing.transforms import get_eval_transforms
from src.settings.config import DEVICE, LABEL_ENCODER_PATH, MODEL_PATH
from src.utils.label_encoder import decode_prediction, load_label_encoder


@lru_cache
def get_inference_model():
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


@lru_cache
def get_label_encoder():
    return load_label_encoder(LABEL_ENCODER_PATH)


def get_eval_transform():
    return get_eval_transforms()


def predict_with_confidence(model, image_tensor, label_encoder):
    """Devuelve indice, etiqueta, confianza y probabilidades por clase."""
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item())
    probabilities = {
        decode_prediction(label_encoder, i): float(probs[i].item())
        for i in range(len(probs))
    }
    label = decode_prediction(label_encoder, pred_idx)
    return pred_idx, label, confidence, probabilities
