"""Descarga DS1 (fall-detection-raskl) y DS2 (fa-nunl5) desde Roboflow."""

import os
import sys
import warnings
from pathlib import Path

import urllib3
from dotenv import load_dotenv
from roboflow import Roboflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.fusion import DS1_SPEC, DS2_SPEC, get_dataset_path  # noqa: E402

load_dotenv(ROOT / ".env")

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("Falta ROBOFLOW_API_KEY en .env")

RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def _create_roboflow() -> Roboflow:
    try:
        return Roboflow(api_key=api_key)
    except Exception as exc:
        if "SSL" not in str(exc) and "certificate" not in str(exc).lower():
            raise
        warnings.warn(
            "Fallo verificacion SSL con Roboflow; reintentando sin verificar certificado.",
            stacklevel=2,
        )
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        import requests

        _orig_request = requests.Session.request

        def _request_no_verify(self, method, url, **kwargs):
            kwargs.setdefault("verify", False)
            return _orig_request(self, method, url, **kwargs)

        requests.Session.request = _request_no_verify
        return Roboflow(api_key=api_key)


rf = _create_roboflow()

print("\n" + "=" * 70)
print("DESCARGA DE DATASETS DESDE ROBOFLOW")
print("=" * 70)

for spec in (DS1_SPEC, DS2_SPEC):
    print(f"\n--- Descargando {spec['prefix']}: {spec['project']} v{spec['version']} ---")
    project = rf.workspace(spec["workspace"]).project(spec["project"])
    version = project.version(spec["version"])
    dataset = version.download("folder", location=str(RAW / spec["subdir"]))
    path = get_dataset_path(dataset)
    print(f"Dataset guardado en: {path}")

print("\nDescarga completa. Siguiente paso: python scripts/fuse_datasets.py")
