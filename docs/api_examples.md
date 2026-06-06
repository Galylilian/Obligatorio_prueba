# Ejemplos de uso de la API — Detector de Caídas

Base URL local: `http://localhost:8080`

## Health check

```bash
curl http://localhost:8080/health
```

Respuesta:

```json
{"status": "ok"}
```

## Predicción CNN

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@/ruta/a/imagen.jpg"
```

Respuesta:

```json
{
  "prediction": 0,
  "label": "fall"
}
```

## Grad-CAM

```bash
curl -X POST http://localhost:8080/gradcam \
  -F "file=@/ruta/a/imagen.jpg" \
  --output gradcam.jpg
```

Devuelve una imagen JPEG con el mapa de activación superpuesto.

## Documentación interactiva

- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## Predicción batch (offline)

```bash
python scripts/batch_predict.py data/fused/test/fall -o data/metadata/batch_predictions.csv
```
