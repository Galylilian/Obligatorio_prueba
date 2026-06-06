"""Utilidades para ejecutar módulos src sin conflictos de PYTHONPATH."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_env(root: Path | None = None) -> dict[str, str]:
    root = root or project_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    return env


def run_module(module: str, root: Path | None = None) -> None:
    root = root or project_root()
    code = (
        f"import sys; sys.path.insert(0, {str(root)!r}); "
        f"from {module} import main; main()"
    )
    subprocess.check_call(
        [sys.executable, "-I", "-c", code],
        cwd=root,
        env=project_env(root),
    )


def run_script(relative_path: str, root: Path | None = None) -> None:
    root = root or project_root()
    cmd = [sys.executable, str(root / relative_path)]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=root, env=project_env(root))
