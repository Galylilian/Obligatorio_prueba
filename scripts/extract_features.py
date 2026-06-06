"""Extrae features tabulares y las une al metadata CSV."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.eda_stats import FEATURE_COLS, metadata_path  # noqa: E402
from src.features.tabular_features import extract_features_for_metadata  # noqa: E402
from src.settings.config import settings  # noqa: E402


def main() -> None:
    meta_path = metadata_path()
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata no encontrado: {meta_path}. Ejecutá python scripts/fuse_datasets.py"
        )

    import pandas as pd

    meta = pd.read_csv(meta_path)
    missing = [c for c in FEATURE_COLS if c not in meta.columns]
    if not missing:
        print("Features tabulares ya presentes en metadata")
        return

    out = Path(settings.metadata_path)
    merged = extract_features_for_metadata(meta_path, output_csv=out)
    print(f"Metadata con features guardado en {out} ({len(merged)} filas)")


if __name__ == "__main__":
    main()
