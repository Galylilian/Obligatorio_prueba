"""Fusión de DS1 + DS2 (Roboflow) y generación de metadata Fall / Not Fall."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.settings.config import settings

VALID_LABELS = frozenset({"fall", "not_fall"})
VALID_SPLITS = frozenset({"train", "valid", "test"})
LABEL_TO_CODE = {"fall": 1, "not_fall": 0}

DS1_SPEC = {
    "workspace": "iot-c5yvo",
    "project": "fall-detection-raskl",
    "version": 2,
    "subdir": "fall-detection-raskl",
    "prefix": "DS1",
}
DS2_SPEC = {
    "workspace": "jay-buvdf",
    "project": "fa-nunl5",
    "version": 1,
    "subdir": "fa-nunl5",
    "prefix": "DS2",
}


def get_dataset_path(dataset_obj) -> str:
    if isinstance(dataset_obj, str):
        return dataset_obj
    if hasattr(dataset_obj, "location"):
        return str(dataset_obj.location)
    if hasattr(dataset_obj, "path"):
        return str(dataset_obj.path)
    raise TypeError(f"No se pudo resolver ruta del dataset: {type(dataset_obj)}")


def _resolve_class_dir(split_dir: Path, label: str) -> Path | None:
    direct = split_dir / label
    if direct.is_dir():
        return direct
    for candidate in split_dir.iterdir():
        if candidate.is_dir() and candidate.name.lower().replace(" ", "_") == label:
            return candidate
    return None


def get_dataset_structure(base_path: str | Path) -> dict:
    base = Path(base_path)
    structure: dict[str, dict[str, list[str]]] = {}
    for split_dir in sorted(base.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name.lower()
        if split_name not in VALID_SPLITS:
            continue
        structure[split_name] = {}
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name.lower().replace(" ", "_")
            if label not in VALID_LABELS:
                continue
            files = [
                f.name
                for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
            structure[split_name][label] = sorted(files)
    return structure


def fuse_datasets(
    src_path: str | Path,
    dst_path: str | Path,
    dataset_prefix: str,
) -> tuple[dict, dict]:
    src = Path(src_path)
    dst = Path(dst_path)
    stats: dict[str, int] = {"copied": 0, "skipped": 0, "ignored_labels": 0}
    structure = get_dataset_structure(src)

    for split, labels in structure.items():
        for label, filenames in labels.items():
            src_class = _resolve_class_dir(src / split, label)
            if src_class is None:
                continue
            dst_class = dst / split / label
            dst_class.mkdir(parents=True, exist_ok=True)
            for fname in filenames:
                prefixed = f"{dataset_prefix}_{fname}"
                dest_file = dst_class / prefixed
                if dest_file.exists():
                    stats["skipped"] += 1
                    continue
                src_file = src_class / fname
                if src_file.exists():
                    shutil.copy2(src_file, dest_file)
                    stats["copied"] += 1

    for split_dir in Path(src_path).glob("*"):
        if split_dir.is_dir() and split_dir.name.lower() in VALID_SPLITS:
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    norm = class_dir.name.lower().replace(" ", "_")
                    if norm not in VALID_LABELS:
                        stats["ignored_labels"] += sum(1 for _ in class_dir.iterdir() if _.is_file())

    return structure, stats


class DatasetFuser:
    def __init__(
        self,
        ds1_path: str | Path | None = None,
        ds2_path: str | Path | None = None,
        fused_dir: Path | None = None,
        metadata_path: Path | None = None,
    ):
        raw = Path(settings.raw_data_dir)
        self.ds1_path = Path(ds1_path or raw / DS1_SPEC["subdir"])
        self.ds2_path = Path(ds2_path or raw / DS2_SPEC["subdir"])
        self.fused_dir = Path(fused_dir or settings.data_dir)
        self.metadata_path = Path(metadata_path or settings.metadata_path)

    def fuse_all(self) -> pd.DataFrame:
        self.fused_dir.mkdir(parents=True, exist_ok=True)
        print(f"Fusionando DS1 desde {self.ds1_path}")
        fuse_datasets(self.ds1_path, self.fused_dir, DS1_SPEC["prefix"])
        print(f"Fusionando DS2 desde {self.ds2_path}")
        fuse_datasets(self.ds2_path, self.fused_dir, DS2_SPEC["prefix"])
        df = self._build_metadata()
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.metadata_path, index=False)
        print(f"Metadata guardada en {self.metadata_path}")
        self._print_balance_summary(df)
        return df

    def _build_metadata(self) -> pd.DataFrame:
        rows = []
        for split in ("train", "valid", "test"):
            split_dir = self.fused_dir / split
            if not split_dir.exists():
                continue
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                label = class_dir.name.lower().replace(" ", "_")
                if label not in VALID_LABELS:
                    continue
                for img_path in class_dir.iterdir():
                    if not img_path.is_file():
                        continue
                    fname = img_path.name
                    if fname.startswith("DS1_"):
                        source = "DS1"
                    elif fname.startswith("DS2_"):
                        source = "DS2"
                    else:
                        source = "unknown"
                    rows.append(
                        {
                            "split": split,
                            "label": label,
                            "label_code": LABEL_TO_CODE[label],
                            "path": str(img_path.resolve()),
                            "filename": fname,
                            "dataset_source": source,
                        }
                    )
        return pd.DataFrame(rows)

    @staticmethod
    def _print_balance_summary(df: pd.DataFrame) -> None:
        print("\n=== Balance por split ===")
        for split in sorted(df["split"].unique()):
            sub = df[df["split"] == split]
            n_fall = (sub["label"] == "fall").sum()
            n_not = (sub["label"] == "not_fall").sum()
            ratio = n_fall / max(n_not, 1)
            print(f"  {split}: fall={n_fall}, not_fall={n_not}, ratio={ratio:.3f}")


def check_leakage_by_imagename(metadata_df: pd.DataFrame) -> list[str]:
    def base_name(fname: str) -> str:
        for prefix in ("DS1_", "DS2_"):
            if fname.startswith(prefix):
                return fname[len(prefix):]
        return fname

    df = metadata_df.copy()
    df["base"] = df["filename"].map(base_name)
    train_bases = set(df[df["split"] == "train"]["base"])
    test_bases = set(df[df["split"] == "test"]["base"])
    return sorted(train_bases & test_bases)[:20]
