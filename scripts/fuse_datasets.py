"""Fusiona DS1 + DS2 en data/fused y genera metadata CSV."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.fusion import DatasetFuser  # noqa: E402


def main() -> None:
    fuser = DatasetFuser()
    df = fuser.fuse_all()
    print(f"\n{len(df)} registros en metadata")


if __name__ == "__main__":
    main()
