# Arquitectura — Fall Detector

Explicación técnica de los componentes del sistema y de qué es responsable cada archivo. Para el detalle de requests/responses de la API ver [endpoints.md](endpoints.md); para el paso a paso de instalación ver el [README](../README.md).

---

## Vista general

El sistema tiene dos pipelines independientes que comparten el mismo modelo:

- **Offline (entrenamiento)**: construye el dataset etiquetado y entrena/evalúa el modelo. Se corre a mano, fuera de Docker.
- **Online (producción)**: sirve el modelo entrenado vía API REST, expone una UI y persiste cada predicción para analítica.

```text
┌─────────────────────────── OFFLINE ───────────────────────────┐
│ scrape_dataset.py / extract_video_frames.py → data/raw/pool/  │
│         label_tool.py (etiquetado manual) → fall/ no_fall/    │
│         convert_dataset.py → data/processed/ (train/valid/test)│
│         train.py → models/resnet18*.pth                       │
│         evaluate.py → metrics.json                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (modelo + métricas)
┌─────────────────────────── ONLINE ────────────────────────────┐
│  Streamlit (8501) ──HTTP──▶ FastAPI (8080) ──SQLAlchemy──▶ Postgres (5432)
│                                   │                              │
│                                   ▼                              │
│                            Grafana (3000) ◀───────── lee Postgres directamente
└─────────────────────────────────────────────────────────────┘
```

Todos los servicios online se orquestan con `docker-compose.yml` (ver sección [Despliegue](#despliegue-docker-compose)).

---

## Capas de `src/`

### `src/api/` — API REST (FastAPI)

- **`app.py`** — Punto de entrada de la aplicación. Registra los routers, y en el `lifespan` llama a `init_db()` para crear las tablas en Postgres al arrancar el contenedor. También parchea el esquema OpenAPI (`custom_openapi`) para que Swagger renderice correctamente los campos `list[UploadFile]` como selector de archivos (FastAPI/OpenAPI 3.1 los describe con `contentMediaType`, que Swagger UI no soporta en arrays).
- **`routers/health.py`** — `GET /health`, usado como healthcheck.
- **`routers/predict.py`** — `POST /predict` y `POST /predict/batch`. Carga el `ImageClassifier` (singleton), corre inferencia y guarda cada resultado en la tabla `predictions`.
- **`routers/gradcam.py`** — `POST /gradcam`. Genera el heatmap de explicabilidad y devuelve la imagen (`image/jpeg`).
- **`routers/predict_video.py`** — `POST /predict/video`. Delega en `detect_falls_from_video()` (`src/utils/video_detection.py`) el muestreo de frames y la inferencia.
- **`routers/dashboard.py`** — `GET /dashboard/stats`. Combina agregados de Postgres (conteos, distribución de labels) con las métricas offline de `metrics.json`.

### `src/core/` — Lógica de Machine Learning

- **`model.py`** — Define la arquitectura: ResNet18 preentrenada en ImageNet con la capa `fc` final reemplazada por una capa lineal de 2 clases (fine-tuning completo, no solo la última capa).
- **`train.py`** — Entrenamiento: calcula `class weights` desde la distribución real de `data/processed/train/`, entrena con `CrossEntropyLoss` ponderado, guarda el mejor checkpoint (`models/resnet18.pth`) y genera además la versión cuantizada (`models/resnet18_quantized.pth`) con `torch.ao.quantization.quantize_dynamic`.
- **`evaluate.py`** — Evaluación offline sobre `data/processed/test/`. Genera `metrics.json` con accuracy, precision, recall, F1 y matriz de confusión (usa `src/utils/metrics.py`).
- **`classification.py`** — `ImageClassifier`: carga el modelo (`MODEL_PATH`/`MODEL_TYPE` desde `src/settings/config.py`), aplica `get_test_transforms()` y expone el método de inferencia usado por todos los routers. Es el único punto donde se decide qué pesos (`normal` o `quantized`) sirve la API.
- **`gradcam.py`** — Implementación de GradCAM con forward/backward hooks sobre `layer4[-1]` de ResNet18 (última capa convolucional), para producir el mapa de activación.
- **`preprocessing/transforms.py`** — Único lugar donde se definen las transformaciones de imagen: `get_train_transforms()` (con augmentation) y `get_test_transforms()` (determinística). La API usa exactamente `get_test_transforms()`, igual que la evaluación offline.

### `src/data/`

- **`dataset.py`** — `get_dataloaders()`: construye los `DataLoader` de train/valid/test a partir de `data/processed/` usando `torchvision.datasets.ImageFolder`.

### `src/db/` — Persistencia

- **`database.py`** — Engine y sesión de SQLAlchemy (`DATABASE_URL`), `Base` declarativa, `get_db()` (dependency de FastAPI) e `init_db()` (crea las tablas si no existen).
- **`models.py`** — Tabla `predictions`: `id`, `filename`, `label`, `confidence`, `model_type`, `created_at`. Cada llamada a `/predict`, `/predict/batch` y `/predict/video` inserta una fila acá; es la fuente de datos tanto de `/dashboard/stats` como de los paneles de Grafana.

### `src/settings/config.py`

Configuración centralizada leída de variables de entorno: `MODEL_TYPE` / `MODEL_PATH` (selección normal vs. cuantizado), `DEVICE` (`cuda` si hay GPU, si no `cpu`) y `DATABASE_URL`.

### `src/utils/`

- **`metrics.py`** — `compute_metrics()`: accuracy, precision, recall, F1, matriz de confusión (usado por `evaluate.py`).
- **`logger.py`** — `get_logger()`, logging centralizado.
- **`video_detection.py`** — `detect_falls_from_video()`: recorre el video frame a frame con OpenCV, analiza un frame cada 5 segundos (para evitar redundancia), corre el clasificador sobre cada uno y guarda en disco (`data/video/frames/`) los frames donde detectó `fall`.

---

## Frontend — `app/streamlit_app.py`

UI en Streamlit que consume la API por HTTP (no importa código de `src/` directamente). Tres vistas principales:

- **Dashboard**: consume `GET /dashboard/stats` para mostrar contadores operacionales y métricas del modelo.
- **Predicción de imágenes**: sube archivos a `POST /predict` / `/predict/batch` y `/gradcam`; muestra una advertencia si `confidence < 0.7`.
- **Predicción de video**: sube el archivo a `POST /predict/video` y muestra el resumen de frames analizados.

---

## Observabilidad — Grafana

`grafana/provisioning/` deja Grafana pre-configurado al levantar `docker-compose`:

- **`datasources/postgres.yml`** — conexión automática a la misma base Postgres que usa la API (Grafana lee la tabla `predictions` directamente, sin pasar por FastAPI).
- **`dashboards/dashboard.yml`** + **`dashboards/fall_detector.json`** — dashboard con 7 paneles ya armado, provisto automáticamente al iniciar el contenedor.

---

## Despliegue (docker-compose)

`docker-compose.yml` define 4 servicios:

| Servicio | Imagen/Build | Puerto | Depende de |
|---|---|---|---|
| `postgres` | `postgres:15-alpine` | 5432 | — |
| `fastapi` | `Dockerfile` | 8080 | `postgres` (healthy) |
| `streamlit` | `Dockerfile.streamlit` | 8501 | `fastapi` |
| `grafana` | `grafana/grafana:10.4.0` | 3000 | `postgres` |

Puntos relevantes:

- `fastapi` monta `./models` y `./metrics.json` como volúmenes, así que el modelo entrenado se puede reemplazar sin reconstruir la imagen.
- `MODEL_TYPE` se define como variable de entorno del servicio `fastapi`, controlando si sirve `resnet18.pth` o `resnet18_quantized.pth` (ver `src/settings/config.py`).
- `requirements.txt` instala PyTorch en su build CPU-only (`torch==2.7.1+cpu`), pensado para correr en EC2 sin GPU (ver README, sección "Despliegue en AWS").

---

## Decisiones de diseño clave

### Data Leakage
El split train/valid/test se hace una única vez en `convert_dataset.py`, con semilla fija (`42`), antes de cualquier entrenamiento. Ninguna imagen puede aparecer en más de un split.

### Training-Serving Skew
Las transformaciones de test están definidas en un único lugar (`get_test_transforms()` en `src/core/preprocessing/transforms.py`) y son usadas tanto por `evaluate.py` como por `ImageClassifier` en producción, evitando divergencias entre el preprocesamiento offline y el online.

### Desbalance de clases
`train.py` calcula `class weights` inversamente proporcionales a la frecuencia de cada clase directamente desde el dataset de train (sin configuración manual), penalizando más los errores sobre la clase minoritaria y crítica (`fall`).

### Quantization dinámica
`scripts/benchmark_quantization.py` mide la latencia real de `resnet18.pth` vs. `resnet18_quantized.pth` (→ `benchmark_quantization.json`). Conclusión: no mejora la latencia porque `torch.ao.quantization.quantize_dynamic` sólo cuantiza capas `Linear` (la única en ResNet18 es la `fc` final, minúscula frente al backbone convolucional). El detalle completo está en el README, sección "Modelos".

`scripts/compare_models.py` complementa ese benchmark comparando **predicciones** en vez de latencia: carga ambos modelos localmente (sin pasar por la API), corre inferencia sobre `data/processed/test/` imagen por imagen y reporta el % de acuerdo entre `normal` y `quantized` (→ `compare_models.json`). Sirve para confirmar que la cuantización no degrada la calidad del modelo, aunque tampoco mejore su latencia.

---

## Tests — `tests/`

- **`conftest.py`** — fixtures compartidos, incluyendo un cliente de test de FastAPI con una base SQLite de archivo (`test.db`) en lugar de Postgres.
- **`test_api.py`** / **`api/test_routers.py`** — tests de integración sobre `/predict`, `/gradcam` y `/dashboard/stats`.
