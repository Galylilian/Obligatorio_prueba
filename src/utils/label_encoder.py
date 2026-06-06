"""Persistencia del mapeo índice → etiqueta de clase."""

from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.preprocessing import LabelEncoder


def save_label_encoder(class_names: list[str], path: str | Path) -> None:
    encoder = LabelEncoder()
    encoder.fit(class_names)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(encoder, f)


def load_label_encoder(path: str | Path) -> LabelEncoder:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def decode_prediction(encoder: LabelEncoder, class_index: int) -> str:
    return encoder.inverse_transform([class_index])[0]
