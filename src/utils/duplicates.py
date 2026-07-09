import pathlib

import numpy as np
from PIL import Image

DEFAULT_HAMMING_THRESHOLD = 4  # sobre 64 bits de dHash


def dhash(image: Image.Image, hash_size: int = 8) -> np.ndarray:
    """Perceptual hash (dHash): compara pixeles adyacentes tras reducir la
    imagen a hash_size+1 x hash_size en escala de grises. Casero (sin
    dependencia de `imagehash`) porque son pocas lineas sobre PIL puro.
    """
    small = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def build_duplicate_groups(
    image_dir: pathlib.Path,
    filenames: set[str],
    threshold: int = DEFAULT_HAMMING_THRESHOLD,
) -> dict[str, str]:
    """filename -> id de su grupo de casi-duplicados (distancia de Hamming <= threshold).

    Frames de video separados por poco tiempo (o fotos casi identicas) quedan
    en el mismo grupo. Usado por convert_dataset.py (para que un grupo entero
    vaya al mismo split) y por scripts/find_inconsistent_duplicates.py (para
    detectar labels en conflicto dentro de un mismo grupo).
    """
    names = sorted(filenames)
    hashes = {name: dhash(Image.open(image_dir / name)) for name in names}

    parent = {name: name for name in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(names)):
        hi = hashes[names[i]]
        for j in range(i + 1, len(names)):
            if np.count_nonzero(hi != hashes[names[j]]) <= threshold:
                union(names[i], names[j])

    return {name: find(name) for name in names}
