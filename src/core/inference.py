"""Carga lazy del modelo y artefactos para inferencia (API)."""

from __future__ import annotations

from functools import lru_cache

import torch

from src.core.model import get_model
from src.preprocessing.transforms import get_eval_transforms
from src.settings.config import DEVICE, LABEL_ENCODER_PATH, MODEL_PATH
from src.utils.label_encoder import load_label_encoder


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
