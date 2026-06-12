# Documentación de Endpoints — Fall Detector API

Base URL local: `http://localhost:8080`  
Documentación interactiva (Swagger): `http://localhost:8080/docs`

---

## Índice

- [GET /health](#get-health)
- [POST /predict](#post-predict)
- [POST /predict (batch)](#post-predict-batch)
- [POST /gradcam](#post-gradcam)
- [POST /predict/video](#post-predictvideo)
- [GET /dashboard/stats](#get-dashboardstats)

---

## GET /health

Verifica que la API está funcionando correctamente.

**Request**
```
GET /health
```

**Response**
```json
{
  "status": "ok"
}
```

---

## POST /predict

Clasifica una imagen como `fall` (caída) o `no_fall` (no caída).

**Request**

| Campo | Tipo | Descripción |
|---|---|---|
| `file` | `UploadFile` | Imagen en formato JPG o PNG |

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@imagen.jpg"
```

**Response**
```json
{
  "label": "fall",
  "confidence": 0.9823
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `label` | `string` | Clase predicha: `fall` o `no_fall` |
| `confidence` | `float` | Confianza del modelo entre 0 y 1 |

**Notas**
- Cada llamada a este endpoint registra la predicción en la base de datos PostgreSQL con timestamp, label, confianza y tipo de modelo usado.
- Si la confianza es menor a 0.7, Streamlit muestra una advertencia de baja confianza.

---

## POST /predict (batch)

Clasifica múltiples imágenes en una sola llamada enviando los archivos como `multipart/form-data`.

**Request**

```bash
curl -X POST http://localhost:8080/predict/batch \
  -F "files=@imagen1.jpg" \
  -F "files=@imagen2.jpg" \
  -F "files=@imagen3.jpg"
```

**Response**
```json
{
  "results": [
    {
      "filename": "imagen1.jpg",
      "label": "fall",
      "confidence": 0.9823
    },
    {
      "filename": "imagen2.jpg",
      "label": "no_fall",
      "confidence": 0.8741
    },
    {
      "filename": "imagen3.jpg",
      "label": "fall",
      "confidence": 0.6512
    }
  ],
  "total": 3,
  "inference_time": 0.4821
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `results` | `list` | Lista de predicciones por imagen |
| `filename` | `string` | Nombre del archivo enviado |
| `label` | `string` | Clase predicha: `fall` o `no_fall` |
| `confidence` | `float` | Confianza del modelo entre 0 y 1 |
| `total` | `int` | Cantidad de imágenes procesadas |
| `inference_time` | `float` | Tiempo total de inferencia en segundos |

---

## POST /gradcam

Genera un mapa de calor GradCAM sobre la imagen indicando las regiones que el modelo consideró más relevantes para la predicción.

**Request**

| Campo | Tipo | Descripción |
|---|---|---|
| `file` | `UploadFile` | Imagen en formato JPG o PNG |

```bash
curl -X POST http://localhost:8080/gradcam \
  -F "file=@imagen.jpg" \
  --output gradcam_result.jpg
```

**Response**

Devuelve directamente la imagen con el heatmap superpuesto en formato `image/jpeg`.

**Notas**
- El heatmap se genera sobre `layer4[-1]` de ResNet18, la última capa convolucional.
- Las zonas en rojo indican las regiones con mayor peso en la decisión del modelo.
- Se aplica el mismo preprocesamiento que en inferencia normal (`get_test_transforms()`).

---

## POST /predict/video

Procesa un video frame a frame y detecta caídas. Analiza un frame cada 5 segundos.

**Request**

| Campo | Tipo | Descripción |
|---|---|---|
| `file` | `UploadFile` | Video en formato MP4 |

```bash
curl -X POST http://localhost:8080/predict/video \
  -F "file=@video.mp4"
```

**Response**
```json
{
  "message": "Video procesado correctamente",
  "total_frames": 12,
  "falls_detected": 3,
  "preview": [
    {
      "frame": 0,
      "time_sec": 0.0,
      "label": "no_fall",
      "confidence": 0.9123,
      "is_fall": false
    },
    {
      "frame": 125,
      "time_sec": 5.0,
      "label": "fall",
      "confidence": 0.8754,
      "is_fall": true
    }
  ]
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `total_frames` | `int` | Cantidad de frames analizados |
| `falls_detected` | `int` | Cantidad de frames con caída detectada |
| `preview` | `list` | Primeros 20 resultados del análisis |
| `time_sec` | `float` | Segundo del video correspondiente al frame |
| `is_fall` | `bool` | `true` si el modelo clasificó como caída |

**Notas**
- Los frames donde se detecta una caída se guardan como imágenes en `data/video/frames/`.
- El intervalo de análisis es de 5 segundos para evitar redundancia en videos largos.

---

## GET /dashboard/stats

Devuelve estadísticas operacionales del sistema y métricas del modelo entrenado.

**Request**
```
GET /dashboard/stats
```

```bash
curl http://localhost:8080/dashboard/stats
```

**Response**
```json
{
  "total_predictions": 142,
  "classified_today": 23,
  "falls_today": 7,
  "falls_week": 31,
  "label_distribution": {
    "fall": 58,
    "no_fall": 84
  },
  "high_risk_persons": 7,
  "analytics_enabled": true,
  "model": {
    "splits": {
      "valid": {
        "accuracy": 0.9397,
        "precision": 0.8929,
        "recall": 0.9804,
        "f1_score": 0.9346,
        "confusion_matrix": [[59, 6], [1, 50]]
      }
    }
  }
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `total_predictions` | `int` | Total histórico de predicciones registradas |
| `classified_today` | `int` | Imágenes clasificadas en el día de hoy |
| `falls_today` | `int` | Caídas detectadas hoy |
| `falls_week` | `int` | Caídas detectadas en los últimos 7 días |
| `label_distribution` | `object` | Distribución de clases en el historial |
| `model.splits.valid` | `object` | Métricas del modelo evaluadas sobre el test set |

**Notas**
- Los contadores (`falls_today`, `classified_today`, etc.) se calculan en tiempo real desde PostgreSQL.
- Las métricas del modelo (`accuracy`, `f1_score`, etc.) se leen desde `metrics.json` generado por `evaluate.py`.
- Este endpoint alimenta tanto el dashboard de Streamlit como Grafana.

---

## Errores comunes

| Código | Descripción |
|---|---|
| `200` | Predicción exitosa |
| `422` | Error de validación — revisar formato del archivo enviado |
| `500` | Error interno — revisar logs del contenedor con `docker logs caida-detector-api` |

Para ver los logs en tiempo real:

```bash
docker logs -f caida-detector-api
```