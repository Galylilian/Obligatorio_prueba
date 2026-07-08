"""
Compara la latencia de inferencia (CPU) entre el modelo normal (resnet18.pth)
y el modelo cuantizado (resnet18_quantized.pth), usando imágenes reales del
test_loader (batch_size=1, igual que /predict en la API).

La cuantización dinámica de PyTorch solo acelera inferencia en CPU, por eso
el benchmark fuerza device="cpu" para ambos modelos.
"""

import io
import json
import pathlib
import statistics
import time

import torch

from src.core.model import get_model
from src.data.dataset import get_dataloaders
from src.utils.logger import get_logger

logger = get_logger("benchmark_quantization")

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "resnet18.pth"
QUANTIZED_MODEL_PATH = MODELS_DIR / "resnet18_quantized.pth"

WARMUP_BATCHES = 10


def load_normal_model():
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def load_quantized_model():
    model = get_model(pretrained=False)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    quantized_model.load_state_dict(torch.load(QUANTIZED_MODEL_PATH, map_location="cpu"))
    quantized_model.eval()
    return quantized_model


def load_quantized_model_extended():
    """
    Intenta ampliar la cuantización dinámica a Conv2d además de Linear.
    NOTA: torch.ao.quantization.quantize_dynamic no soporta Conv2d en su
    mapping por defecto (solo Linear/LSTM/GRU/RNN), por lo que las capas
    convolucionales quedan sin tocar aunque se las incluya en el set.
    Se deja este loader para dejar constancia empírica de esa limitación.
    """
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8,
    )
    quantized_model.eval()
    return quantized_model


def benchmark(model, test_loader):
    latencies_ms = []

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.cpu()

            start = time.perf_counter()
            model(images)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if i < WARMUP_BATCHES:
                continue  # descartar warm-up

            latencies_ms.append(elapsed_ms)

    return latencies_ms


def model_size_mb(model_path=None, model=None):
    if model_path is not None:
        return model_path.stat().st_size / (1024 * 1024)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 * 1024)


def summarize(name, latencies_ms, size_mb):
    mean = statistics.mean(latencies_ms)
    stdev = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    p95 = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]

    logger.info(
        f"{name}: mean={mean:.2f} ms | stdev={stdev:.2f} ms | "
        f"p95={p95:.2f} ms | size={size_mb:.2f} MB | n={len(latencies_ms)}"
    )

    return {
        "mean_ms": mean,
        "stdev_ms": stdev,
        "p95_ms": p95,
        "size_mb": size_mb,
        "n_samples": len(latencies_ms),
    }


def main():
    # batch_size=1 para medir latencia por imagen, igual que /predict
    _, _, test_loader = get_dataloaders(batch_size=1)

    logger.info("Benchmarking modelo normal...")
    normal_model = load_normal_model()
    normal_latencies = benchmark(normal_model, test_loader)
    normal_stats = summarize("Normal (fp32)", normal_latencies, model_size_mb(model_path=MODEL_PATH))

    logger.info("Benchmarking modelo cuantizado (Linear)...")
    quantized_model = load_quantized_model()
    quantized_latencies = benchmark(quantized_model, test_loader)
    quantized_stats = summarize(
        "Cuantizado (int8, Linear)", quantized_latencies, model_size_mb(model_path=QUANTIZED_MODEL_PATH)
    )

    logger.info("Benchmarking modelo cuantizado (Linear + Conv2d)...")
    extended_model = load_quantized_model_extended()
    extended_latencies = benchmark(extended_model, test_loader)
    extended_stats = summarize(
        "Cuantizado (int8, Linear+Conv2d)", extended_latencies, model_size_mb(model=extended_model)
    )

    speedup = normal_stats["mean_ms"] / quantized_stats["mean_ms"]
    size_reduction = 1 - (quantized_stats["size_mb"] / normal_stats["size_mb"])

    speedup_extended = normal_stats["mean_ms"] / extended_stats["mean_ms"]
    size_reduction_extended = 1 - (extended_stats["size_mb"] / normal_stats["size_mb"])

    logger.info(f"Speedup (Linear): {speedup:.2f}x | Reducción de tamaño: {size_reduction*100:.1f}%")
    logger.info(
        f"Speedup (Linear+Conv2d): {speedup_extended:.2f}x | "
        f"Reducción de tamaño: {size_reduction_extended*100:.1f}%"
    )

    results = {
        "normal": normal_stats,
        "quantized_linear": quantized_stats,
        "quantized_linear_conv2d": extended_stats,
        "speedup_linear": speedup,
        "size_reduction_pct_linear": size_reduction * 100,
        "speedup_linear_conv2d": speedup_extended,
        "size_reduction_pct_linear_conv2d": size_reduction_extended * 100,
        "note": (
            "PyTorch (torch.ao.quantization.quantize_dynamic) no soporta "
            "Conv2d en su mapping por defecto: solo Linear/LSTM/GRU/RNN. "
            "Por eso la variante Linear+Conv2d da resultados equivalentes "
            "a la variante solo-Linear (las capas Conv2d quedan sin cuantizar)."
        ),
    }

    output_path = BASE_DIR / "benchmark_quantization.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"✅ Resultados guardados en: {output_path}")


if __name__ == "__main__":
    main()
