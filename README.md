# Fall Detector 🏥

Sistema de Machine Learning para clasificación binaria de caídas en imágenes y video. Detecta si una persona está caída o no utilizando una ResNet18 con fine-tuning, expuesta mediante una API REST con FastAPI y una interfaz visual con Streamlit.

---

## Estructura del proyecto

```
Obligatorio_prueba/
│
├── data/                             # Dataset (NO subir a GitHub)
│   ├── raw/                          # Dataset original descargado de Roboflow
│   │   ├── ds1/
│   │   └── ds2/
│   └── processed/                    # Dataset convertido a clasificación binaria
│       ├── train/
│       │   ├── fall/
│       │   └── no_fall/
│       ├── valid/
│       └── test/
│
├── models/                           # Modelos entrenados
│   ├── resnet18.pth                  # Modelo principal
│   └── resnet18_quantized.pth        # Modelo cuantizado (menor tamaño, menor latencia)
│
├── notebooks/
│   └── eda.ipynb                     # Análisis exploratorio del dataset
│
├── scripts/
│   ├── download_dataset.py           # Descarga dataset desde Roboflow
│   ├── convert_dataset.py            # Convierte formato folder → clasificación binaria
│   └── compare_models.py             # Compara rendimiento de modelos
│
├── src/
│   ├── api/
│   │   ├── app.py                    # Aplicación FastAPI principal
│   │   └── routers/
│   │       ├── predict.py            # Endpoint de predicción (online y batch)
│   │       ├── gradcam.py            # Endpoint de explicabilidad GradCAM
│   │       ├── predict_video.py      # Endpoint de predicción sobre video
│   │       ├── dashboard.py          # Endpoint de métricas y estadísticas
│   │       └── health.py             # Endpoint de health check
│   │
│   ├── core/
│   │   ├── model.py                  # Definición ResNet18
│   │   ├── train.py                  # Entrenamiento del modelo
│   │   ├── evaluate.py               # Evaluación offline con métricas
│   │   ├── classification.py         # Lógica de inferencia
│   │   └── gradcam.py                # Implementación GradCAM
│   │
│   ├── core/preprocessing/
│   │   └── transforms.py             # Transformaciones de imágenes (train y test)
│   │
│   ├── data/
│   │   └── dataset.py                # Carga de datos con ImageFolder
│   │
│   ├── db/
│   │   ├── database.py               # Conexión SQLAlchemy a PostgreSQL
│   │   └── models.py                 # Tabla de predicciones
│   │
│   ├── settings/
│   │   └── config.py                 # Configuración (modelo, dispositivo, DB)
│   │
│   └── utils/
│       ├── metrics.py                # Cálculo de métricas ML
│       ├── logger.py                 # Logger centralizado
│       ├── files.py                  # Manejo de archivos temporales
│       └── video_detection.py        # Procesamiento de video frame a frame
│
├── app/
│   └── streamlit_app.py              # Interfaz visual (Streamlit)
│
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── postgres.yml          # Conexión automática a PostgreSQL
│       └── dashboards/
│           ├── dashboard.yml         # Configuración del proveedor de dashboards
│           └── fall_detector.json    # Dashboard pre-configurado
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── api/
│   │   └── test_routers.py
│   └── core/
│       ├── test_model.py
│       ├── test_inference.py
│       ├── test_gradcam.py
│       └── test_preprocessor.py
│
├── docs/
│   └── endpoints.md                  # Documentación de endpoints
│
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
├── dev-requirements.txt
├── .env                              # Variables de entorno (NO subir a GitHub)
└── .gitignore
```

---

## Problema

Clasificación binaria: dada una imagen o un video, el sistema determina si la persona está **caída** (`fall`) o **no caída** (`no_fall`).

El dataset fue obtenido desde Roboflow en formato de detección de objetos y convertido a un problema de clasificación binaria mediante `convert_dataset.py`, donde cada imagen queda asignada a una de las dos clases según las anotaciones originales.

---

## Modelos

| Modelo | Descripción |
|---|---|
| `resnet18.pth` | ResNet18 con fine-tuning completo sobre todas las capas |
| `resnet18_quantized.pth` | Versión cuantizada (dynamic quantization sobre capas Linear) — menor tamaño y latencia |

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
│   Roboflow Dataset  │
│ (imágenes + labels) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ download_dataset.py │
│  (descarga ds1/ds2) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  convert_dataset.py │
│ (folder → fall /    │
│       no_fall)      │
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
│  (ResNet18 fine-    │
│      tuning)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  models/*.pth       │
│ (normal/quantized)  │
└─────────────────────┘


Pipeline de producción (online)

┌──────────────────────────┐
│   docker-compose up      │
│ FastAPI + Streamlit      │
│ + PostgreSQL + Grafana   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│       FastAPI API        │
│     (src/api/app.py)     │
└────┬──────────┬──────────┘
     │          │          │
     ▼          ▼          ▼
/predict   /gradcam   /predict/video
     │
     ▼
PostgreSQL (predicciones)
     │
     ▼
Grafana (dashboard operacional)
```

---

## Desafíos de producción resueltos

### Data Leakage
Se evitó mediante separación estricta de los conjuntos `train`, `valid` y `test` desde el paso de conversión del dataset. Ninguna imagen del conjunto de test es vista durante el entrenamiento.

### Training-Serving Skew
Se unificaron las transformaciones de inferencia en un módulo compartido (`transforms.py`). El endpoint de la API usa exactamente las mismas transformaciones (`get_test_transforms()`) que se aplican al conjunto de test durante la evaluación offline.

---

## Requerimientos electivos implementados

| Electivo | Implementación |
|---|---|
| ✅ Explicabilidad | GradCAM sobre la última capa convolucional de ResNet18 |
| ✅ Visualización | Streamlit con predicción de imágenes, video y dashboard |
| ✅ Optimización de modelos | Quantization dinámica (qint8) + Data Augmentation (flip, rotación) |

---

## Cómo correr el proyecto

### 1. Configurar variables de entorno

Crear un archivo `.env` en la raíz del proyecto:

```env
ROBOFLOW_API_KEY=tu_api_key_aqui
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar y preparar el dataset

```bash
python scripts/download_dataset.py
python scripts/convert_dataset.py
```

### 4. Entrenar el modelo

```bash
python -m src.core.train
```

### 5. Evaluar el modelo

```bash
python -m src.core.evaluate
```

Genera `metrics.json` con accuracy, precision, recall, F1 y matriz de confusión.

### 6. Levantar en producción con Docker

```bash
docker-compose up --build
```

Servicios disponibles:

| Servicio | URL |
|---|---|
| FastAPI (Swagger) | http://localhost:8080/docs |
| Streamlit | http://localhost:8501 |
| Grafana | http://localhost:3000 (admin / admin) |

### 7. Detener los servicios

```bash
docker-compose down -v
```

### 8. Correr tests

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