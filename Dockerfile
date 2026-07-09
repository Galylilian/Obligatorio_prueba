# Use the official Python image as base
FROM python:3.11-slim

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Directorio de trabajo
WORKDIR /app

# Dependencias del sistema (para PyTorch, OpenCV, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt /app/

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY src ./src
COPY models ./models

# Pre-cachear los pesos COCO del detector de personas (ssdlite320_mobilenet_v3_large,
# ~14 MB) durante el build, para que el contenedor no dependa de internet en
# runtime (mismo principio que MODEL_PATH: nada se descarga al arrancar la API).
RUN python -c "from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights as W; ssdlite320_mobilenet_v3_large(weights=W.DEFAULT)"

# Exponer puerto
EXPOSE 8080

# Comando para correr FastAPI correctamente
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]