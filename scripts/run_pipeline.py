"""Pipeline offline: download -> fuse -> features -> train -> evaluate."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.runner import run_module, run_script  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Fall Detection")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        run_script("scripts/download_dataset.py", ROOT)
        run_script("scripts/fuse_datasets.py", ROOT)
        run_script("scripts/extract_features.py", ROOT)
    else:
        run_script("scripts/fuse_datasets.py", ROOT)
        run_script("scripts/extract_features.py", ROOT)

    if not args.skip_train:
        run_module("src.core.train", ROOT)
        run_module("src.core.evaluate", ROOT)

    print("\nPipeline finalizado")


if __name__ == "__main__":
    main()
