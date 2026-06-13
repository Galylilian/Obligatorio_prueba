# Fall Detector 🏥

Sistema de Machine Learning para clasificación binaria de caídas en imágenes y video. Detecta si una persona está caída o no utilizando una ResNet18 con fine-tuning, expuesta mediante una API REST con FastAPI y una interfaz visual con Streamlit.

---

## Contenido del proyecto

```
Obligatorio_prueba/
│
├── data/                                      # Dataset (NO subir a GitHub)
│   ├── raw/                                   # Imágenes scrapeadas sin procesar
│   │   └── scraped/                           # Descargadas por scrape_dataset.py
│   │       ├── fall/                          # Imágenes de caídas
│   │       ├── no_fall/                       # Imágenes sin caída
│   │       └── scraping_log.csv               # Log de cada imagen: query, url, hash, timestamp
│   ├── processed/                             # Dataset listo para entrenamiento
│   │   ├── train/
│   │   │   ├── fall/
│   │   │   └── no_fall/
│   │   ├── valid/
│   │   │   ├── fall/
│   │   │   └── no_fall/
│   │   ├── test/
│   │   │   ├── fall/
│   │   │   └── no_fall/
│   │   └── dataset_labels.csv                 # Trazabilidad completa: imagen, label, split, query
│   └── video/                                 # Videos subidos y frames con caídas
│       ├── uploads/
│       └── frames/
│
├── models/                                    # Modelos entrenados (NO subir a GitHub)
│   ├── resnet18.pth                           # Modelo principal (float32)
│   └── resnet18_quantized.pth                 # Modelo cuantizado (int8, menor latencia)
│
├── notebooks/
│   └── eda.ipynb                              # Análisis exploratorio del dataset
│
├── scripts/
│   ├── scrape_dataset.py                      # Scraper de imágenes
│   ├── convert_dataset.py                     # Divide scraped → train/valid/test + genera CSV
│   └── compare_models.py                      # Compara predicciones entre modelos
│
├── src/
│   │
│   ├── api/                                   # API REST (FastAPI)
│   │   ├── app.py                             # Aplicación principal, registro de routers e init DB
│   │   └── routers/
│   │       ├── health.py                      # GET /health — healthcheck del contenedor
│   │       ├── predict.py                     # POST /predict — clasificación de imagen
│   │       ├── gradcam.py                     # POST /gradcam — heatmap de explicabilidad
│   │       ├── predict_video.py               # POST /predict/video — análisis de video
│   │       └── dashboard.py                   # GET /dashboard/stats — métricas operacionales
│   │
│   ├── core/                                  # Lógica ML
│   │   ├── model.py                           # Definición ResNet18 con fine-tuning
│   │   ├── train.py                           # Entrenamiento, validación por epoch y quantization
│   │   ├── evaluate.py                        # Evaluación offline → genera metrics.json
│   │   ├── classification.py                  # Clase ImageClassifier para inferencia
│   │   ├── gradcam.py                         # Implementación GradCAM con hooks
│   │   └── preprocessing/
│   │       └── transforms.py                  # Transforms de train (augmentation) y test
│   │
│   ├── data/
│   │   └── dataset.py                         # get_dataloaders() con ImageFolder
│   │
│   ├── db/                                    # Capa de base de datos
│   │   ├── __init__.py
│   │   ├── database.py                        # Conexión SQLAlchemy + get_db() + init_db()
│   │   └── models.py                          # Tabla predictions en PostgreSQL
│   │
│   ├── settings/
│   │   └── config.py                          # MODEL_TYPE, MODEL_PATH, DEVICE, DATABASE_URL
│   │
│   └── utils/
│       ├── metrics.py                         # compute_metrics() — accuracy, F1, precision, recall
│       ├── logger.py                          # get_logger() — logger centralizado
│       ├── files.py                           # Manejo de archivos temporales
│       └── video_detection.py                 # detect_falls_from_video() — frame a frame
│
├── app/
│   └── streamlit_app.py                       # Interfaz visual — dashboard, imágenes y video
│
├── grafana/                                   # Configuración automática de Grafana
│   └── provisioning/
│       ├── datasources/
│       │   └── postgres.yml                   # Conexión automática a PostgreSQL
│       └── dashboards/
│           ├── dashboard.yml                  # Proveedor de dashboards (directorio y refresh)
│           └── fall_detector.json             # Dashboard pre-configurado con 7 paneles
│
├── tests/
│   ├── conftest.py                            # Fixtures — cliente de test con DB en memoria
│   ├── test_api.py                            # Test de docs y health
│   ├── api/
│   │   └── test_routers.py                    # Tests de /predict, /gradcam, /dashboard/stats
│   └── core/
│       ├── test_model.py                      # Tests del modelo ResNet18
│       ├── test_inference.py                  # Tests de ImageClassifier
│       ├── test_gradcam.py                    # Tests de GradCAM
│       └── test_preprocessor.py              # Tests de transforms train vs test
│
├── docs/
│   ├── endpoints.md                           # Documentación de endpoints con ejemplos curl
│   └── arquitectura.md                        # Explicación técnica de cada archivo del proyecto
│
├── Dockerfile                                 # Imagen de la API FastAPI
├── Dockerfile.streamlit                       # Imagen de Streamlit
├── docker-compose.yml                         # Orquestación: FastAPI + Streamlit + PostgreSQL + Grafana
├── requirements.txt                           # Dependencias de producción
├── metrics.json                               # Métricas del modelo (generado por evaluate.py)
├── .env                                       # Variables de entorno (NO subir a GitHub)
├── .gitignore
└── README.md
```

---

## Problema

Clasificación binaria: dada una imagen o un video, el sistema determina si la persona está **caída** (`fall`) o **no caída** (`no_fall`).

El dataset fue construido mediante scraping de imágenes desde DuckDuckGo usando queries específicos por clase. La etiqueta se asigna por **Weak Supervision**: la heurística es el término de búsqueda usado para descargar cada imagen. Todo el proceso queda registrado en `scraping_log.csv` y `dataset_labels.csv` para trazabilidad completa.

---

## Dataset

| Fuente | Método de etiquetado | Clases |
|---|---|---|
| DuckDuckGo (scraping) | Weak Supervision por query | `fall` / `no_fall` |

### Queries utilizados

| Clase | Queries |
|---|---|
| `fall` | `person fallen floor indoors`, `elderly person fall ground`, `person collapsed floor`, `man fallen street`, `person lying floor accident`, `person fell down stairs` |
| `no_fall` | `person standing indoors`, `person walking street`, `person sitting chair`, `elderly person walking cane`, `person upright room`, `people standing office` |

### División del dataset

| Split | Proporción |
|---|---|
| train | 70% |
| valid | 15% |
| test | 15% |

La división es aleatoria con semilla fija (`42`) para garantizar reproducibilidad.

---

## Modelos

| Modelo | Descripción |
|---|---|
| `resnet18.pth` | ResNet18 con fine-tuning completo desde pesos ImageNet |
| `resnet18_quantized.pth` | Versión cuantizada (dynamic quantization int8) — menor tamaño y latencia |

### Selección de modelo en producción

Se controla mediante la variable de entorno `MODEL_TYPE` en el `docker-compose.yml`:

```yaml
- MODEL_TYPE=normal      # usa resnet18.pth (default)
- MODEL_TYPE=quantized   # usa resnet18_quantized.pth
```

---

## Arquitectura del sistema

```
Pipeline de entrenamiento (offline)

┌─────────────────────┐
│   PEXELS y PIXABAY  │
│ (búsqueda imágenes) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ scrape_dataset.py   │
│ (descarga + log CSV)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  data/raw/scraped   │
│  fall/ | no_fall/   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ convert_dataset.py  │
│ (split 70/15/15 +   │
│  dataset_labels.csv)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   data/processed    │
│  (ImageFolder ready)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      train.py       │
│ (ResNet18 fine-     │
│  tuning + best ckpt │
│  + quantization)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  models/*.pth       │
│ (normal/quantized)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    evaluate.py      │
│ (métricas offline → │
│   metrics.json)     │
└─────────────────────┘


Pipeline de producción (online)

┌──────────────────────────────────────┐
│         docker-compose up            │
│  FastAPI + Streamlit                 │
│  + PostgreSQL + Grafana              │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│       FastAPI API        │
│     (src/api/app.py)     │
└───┬──────┬──────┬────────┘
    │      │      │
    ▼      ▼      ▼
/predict /gradcam /predict/video
    │
    ▼
PostgreSQL
(tabla predictions)
    │
    ▼
Grafana
(dashboard operacional)


Interfaz visual

┌──────────────────────────┐
│      Streamlit App       │
│   (app/streamlit_app.py) │
└──────────┬───────────────┘
           │ HTTP requests
           ▼
     FastAPI API
           │
           ▼
┌──────────────────────────┐
│    Resultados mostrados  │
│ ✅ métricas del modelo   │
│ ✅ predicción + GradCAM  │
│ ✅ análisis de video     │
└──────────────────────────┘
```

---

## Desafíos de producción resueltos

### Data Leakage
Se evitó mediante separación estricta de los conjuntos `train`, `valid` y `test` en `convert_dataset.py`. La división se hace con semilla fija antes de cualquier entrenamiento, garantizando que ninguna imagen del test sea vista por el modelo.

### Training-Serving Skew
Se unificaron las transformaciones en un módulo compartido (`transforms.py`). La API usa exactamente `get_test_transforms()` — el mismo preprocesamiento que se aplica al conjunto de test durante la evaluación offline.

---

## Requerimientos electivos implementados

| Electivo | Implementación |
|---|---|
| ✅ Scraper de datos | `scrape_dataset.py` — DuckDuckGo, Weak Supervision, log CSV |
| ✅ Explicabilidad | GradCAM sobre `layer4[-1]` de ResNet18 |
| ✅ Visualización | Streamlit con dashboard, predicción de imágenes y video |
| ✅ Optimización de modelos | Data Augmentation (flip, rotación) + Quantization dinámica (int8) |

---

## Cómo correr el proyecto

### 1. Configurar variables de entorno

Crear `.env` en la raíz:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/falldetector
MODEL_TYPE=normal
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Scrapear imágenes

```bash
python scripts/scrape_dataset.py
```

Descarga imágenes desde PEXELS y PIXABAY a `data/raw/scraped/` organizadas por clase (`fall/` y `no_fall/`). Genera `scraping_log.csv` con cada imagen, su query de origen, URL y hash MD5.

> Objetivo: ~180 imágenes por clase (30 imágenes × 6 queries).

### 4. Convertir y dividir el dataset

```bash
python scripts/convert_dataset.py
```

Divide las imágenes en `train/valid/test` (70/15/15) con semilla fija y genera `data/processed/dataset_labels.csv` con la trazabilidad completa de cada imagen.

### 5. Entrenar el modelo

```bash
python -m src.core.train
```

Entrena ResNet18 con fine-tuning desde pesos ImageNet. Guarda el mejor checkpoint por accuracy de validación y genera automáticamente el modelo cuantizado.

### 6. Evaluar el modelo

```bash
python -m src.core.evaluate
```

Genera `metrics.json` con accuracy, precision, recall, F1 y matriz de confusión sobre el conjunto de test.

### 7. Levantar en producción con Docker

```bash
docker-compose up --build
```

| Servicio | URL |
|---|---|
| FastAPI (Swagger) | http://localhost:8080/docs |
| Streamlit | http://localhost:8501 |
| Grafana | http://localhost:3000 (admin / admin) |

### 8. Detener los servicios

```bash
docker-compose down -v
```

### 9. Correr tests

```bash
pytest tests/
```

---

## Métricas del modelo

| Métrica | Valor |
|---|---|
| Accuracy | 93.97% |
| Precision | 89.29% |
| Recall | 98.04% |
| F1 Score | 93.46% |

Evaluado sobre el conjunto de test con el modelo `resnet18.pth`.