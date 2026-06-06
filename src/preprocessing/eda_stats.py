"""Utilidades reutilizables para el EDA del dataset Fall / Not Fall."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from src.preprocessing.fusion import VALID_LABELS, check_leakage_by_imagename
from src.settings.config import settings

BINARY_CLASS_NAMES = ("fall", "not_fall")
POSITIVE_CLASS = "fall"
SPLITS = ("train", "valid", "test")
IMAGE_SIZE = (settings.image_size, settings.image_size)

FEATURE_COLS = [
    "brightness_mean",
    "brightness_std",
    "contrast_score",
    "aspect_ratio",
    "edge_density",
    "blur_score",
    "top_half_brightness",
    "bottom_half_brightness",
    "vertical_brightness_ratio",
]


@dataclass
class SplitStats:
    split: str
    total_images: int
    fall: int
    not_fall: int


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_root(start: Path | None = None) -> Path:
    candidates = [start, Path.cwd()] if start else [Path.cwd()]
    for base in candidates:
        if base is None:
            continue
        for path in [base, *base.parents]:
            if (path / settings.data_dir / "train" / "fall").exists():
                return path
            if (path / "src" / "preprocessing" / "eda_stats.py").exists():
                return path
    return project_root()


def fused_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / settings.data_dir


def raw_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / settings.raw_data_dir


def metadata_path(root: Path | None = None) -> Path:
    return (root or project_root()) / settings.metadata_path


def ensure_dataset_available(root: Path | None = None) -> Path:
    root = root or project_root()
    train_fall = fused_dir(root) / "train" / "fall"
    if not train_fall.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset fusionado en {fused_dir(root)}. "
            "Ejecutá: python scripts/download_dataset.py && python scripts/fuse_datasets.py"
        )
    return root


def collect_split_stats(root: Path | None = None) -> list[SplitStats]:
    base = fused_dir(root)
    stats: list[SplitStats] = []

    for split in SPLITS:
        split_dir = base / split
        if not split_dir.exists():
            continue
        counts = {label: 0 for label in BINARY_CLASS_NAMES}
        for label in BINARY_CLASS_NAMES:
            class_dir = split_dir / label
            if class_dir.is_dir():
                counts[label] = len(list(class_dir.glob("*.*")))
        total = sum(counts.values())
        stats.append(
            SplitStats(
                split=split,
                total_images=total,
                fall=counts["fall"],
                not_fall=counts["not_fall"],
            )
        )
    return stats


def collect_image_sizes(root: Path | None = None) -> list[tuple[int, int]]:
    base = fused_dir(root)
    sizes: list[tuple[int, int]] = []

    for split in SPLITS:
        split_dir = base / split
        if not split_dir.exists():
            continue
        for label in BINARY_CLASS_NAMES:
            class_dir = split_dir / label
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                with Image.open(img_path) as img:
                    sizes.append(img.size)
    return sizes


def count_processed_classes(root: Path | None = None) -> dict[str, dict[str, int]]:
    base = fused_dir(root)
    result: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        split_dir = base / split
        if not split_dir.exists():
            continue
        result[split] = {}
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir() and class_dir.name in VALID_LABELS:
                result[split][class_dir.name] = len(list(class_dir.glob("*.*")))
    return result


def load_metadata(root: Path | None = None) -> pd.DataFrame:
    path = metadata_path(root)
    if path.exists():
        return pd.read_csv(path)
    return build_metadata_dataframe(root)


def build_metadata_dataframe(root: Path | None = None) -> pd.DataFrame:
    base = fused_dir(root)
    rows = []

    for split in SPLITS:
        split_dir = base / split
        if not split_dir.exists():
            continue
        for label in BINARY_CLASS_NAMES:
            class_dir = split_dir / label
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                fname = img_path.name
                if fname.startswith("DS1_"):
                    source = "DS1"
                elif fname.startswith("DS2_"):
                    source = "DS2"
                else:
                    source = "unknown"
                with Image.open(img_path) as img:
                    width, height = img.size
                rows.append(
                    {
                        "split": split,
                        "label": label,
                        "label_code": 1 if label == "fall" else 0,
                        "path": str(img_path.resolve()),
                        "filename": fname,
                        "dataset_source": source,
                        "width": width,
                        "height": height,
                        "aspect_ratio": height / max(width, 1),
                    }
                )
    return pd.DataFrame(rows)


def collect_source_balance(root: Path | None = None) -> pd.DataFrame:
    df = load_metadata(root)
    if df.empty:
        return df
    return (
        df.groupby(["split", "dataset_source", "label"])
        .size()
        .reset_index(name="count")
    )


def check_split_leakage_by_filename(root: Path | None = None) -> list[dict]:
    base = fused_dir(root)
    split_files: dict[str, set[str]] = {}

    for split in SPLITS:
        split_dir = base / split
        if not split_dir.exists():
            continue
        names: set[str] = set()
        for label in BINARY_CLASS_NAMES:
            class_dir = split_dir / label
            if class_dir.is_dir():
                names.update(p.name for p in class_dir.iterdir() if p.is_file())
        split_files[split] = names

    issues = []
    splits = list(split_files.keys())
    for i, split_a in enumerate(splits):
        for split_b in splits[i + 1:]:
            overlap = split_files[split_a] & split_files[split_b]
            if overlap:
                issues.append(
                    {
                        "split_a": split_a,
                        "split_b": split_b,
                        "n_duplicates": len(overlap),
                        "examples": sorted(overlap)[:5],
                    }
                )
    return issues


def check_train_test_leakage(root: Path | None = None) -> list[str]:
    df = load_metadata(root)
    if df.empty:
        return []
    return check_leakage_by_imagename(df)


def get_binary_class_weights(root: Path | None = None) -> list[float]:
    counts = count_processed_classes(root).get("train", {})
    total = sum(counts.values()) or 1
    weights = []
    for class_name in BINARY_CLASS_NAMES:
        count = counts.get(class_name, 1)
        weights.append(total / (len(BINARY_CLASS_NAMES) * count))
    return weights


def split_stats_to_dataframe(stats: list[SplitStats]) -> pd.DataFrame:
    return pd.DataFrame([asdict(s) for s in stats])
