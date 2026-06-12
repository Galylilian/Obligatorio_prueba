import torch
import os

# =============================
# SELECCIÓN DE MODELO
# =============================
# Podés controlar qué modelo usar desde la variable de entorno MODEL_TYPE:
#   MODEL_TYPE=normal     → usa resnet18.pth       (default)
#   MODEL_TYPE=quantized  → usa resnet18_quantized.pth

MODEL_TYPE = os.getenv("MODEL_TYPE", "normal")

if MODEL_TYPE == "quantized":
    MODEL_PATH = os.getenv("MODEL_PATH", "models/resnet18_quantized.pth")
else:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/resnet18.pth")

# =============================
# DISPOSITIVO
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# BASE DE DATOS
# =============================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/falldetector"
)