"""Extracción de features tabulares desde imágenes (9 features del EDA)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.eda_stats import FEATURE_COLS


class TabularFeatureExtractor:
    def extract_from_array(self, img_bgr: np.ndarray) -> dict[str, float]:
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2].astype(np.float32) / 255.0

        brightness_mean = float(np.mean(v_channel))
        brightness_std = float(np.std(v_channel))
        contrast_score = float(np.std(gray.astype(np.float32)))
        aspect_ratio = float(h / max(w, 1))
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / max(edges.size, 1))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        mid = h // 2
        top_half_brightness = float(np.mean(v_channel[:mid, :]))
        bottom_half_brightness = float(np.mean(v_channel[mid:, :]))
        vertical_brightness_ratio = top_half_brightness / (bottom_half_brightness + 1e-6)

        return {
            "brightness_mean": brightness_mean,
            "brightness_std": brightness_std,
            "contrast_score": contrast_score,
            "aspect_ratio": aspect_ratio,
            "edge_density": edge_density,
            "blur_score": blur_score,
            "top_half_brightness": top_half_brightness,
            "bottom_half_brightness": bottom_half_brightness,
            "vertical_brightness_ratio": vertical_brightness_ratio,
        }

    def extract(self, image_path: str | Path) -> dict[str, float]:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"No se pudo leer imagen: {image_path}")
        return self.extract_from_array(img_bgr)

    def extract_batch(self, paths: list[str | Path], n_workers: int = 4) -> pd.DataFrame:
        records: list[dict] = []

        def _task(p: str | Path) -> dict:
            feats = self.extract(p)
            feats["path"] = str(Path(p).resolve())
            return feats

        if n_workers <= 1:
            for p in paths:
                records.append(_task(p))
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_task, p) for p in paths]
                for future in as_completed(futures):
                    records.append(future.result())

        return pd.DataFrame(records)


def extract_features_for_metadata(metadata_csv: Path, output_csv: Path | None = None) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv)
    missing = [c for c in FEATURE_COLS if c not in meta.columns]
    if not missing:
        return meta

    extractor = TabularFeatureExtractor()
    feat_df = extractor.extract_batch(meta["path"].tolist(), n_workers=4)
    merged = meta.merge(feat_df, on="path", how="left")
    if output_csv:
        merged.to_csv(output_csv, index=False)
    return merged
