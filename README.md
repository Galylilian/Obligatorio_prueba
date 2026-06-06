obligatorio/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── fused/
│   └── metadata/
│
├── docs/
│   ├── arquitectura.png
│   ├── api_examples.md
│   └── informe.pdf
│
├── mlruns/
│
├── models/
│   ├── resnet18_best.pth
│   └── label_encoder.pkl
│
├── notebooks/
│   └── EDA_Caidas.ipynb
│
├── scripts/
│   ├── download_dataset.py
│   ├── fuse_datasets.py
│   ├── extract_features.py
│   ├── batch_predict.py
│   └── run_pipeline.py
│
├── src/
│   ├── api/
│   ├── core/
│   ├── preprocessing/
│   ├── features/
│   ├── explainability/
│   ├── settings/
│   └── utils/
│
├── tests/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md


# Detección de caídas — MLP en Producción

## Problema

Clasificación binaria **Fall / Not Fall** usando información visual y metadatos tabulares derivados de las imágenes.

| Clase | Descripción |
|-------|-------------|
| `fall` | Situación de caída |
| `not_fall` | Situación normal |

**Fuentes:** Roboflow DS1 (`fall-detection-raskl` v2) + DS2 (`fa-nunl5` v1).

## Modelos

- CNN ResNet18 (clasificación principal)
- Artefactos: `models/resnet18_best.pth` + `models/label_encoder.pkl`

## API

- `POST /predict` — CNN
- `POST /gradcam` — explicabilidad

Ver ejemplos en [`docs/api_examples.md`](docs/api_examples.md).

## Entornos y dependencias

| Carpeta | Entorno | Dependencias |
|---------|---------|--------------|
| `src/api`, `src/core`, `src/explainability` | Producción | `requirements.txt` |
| `scripts`, `notebooks`, `tests`, `app` | Desarrollo | `requirements-dev.txt` |

```bash
pip install -r requirements-dev.txt   # desarrollo
pip install -r requirements.txt       # producción (Docker)
```

### Variables de entorno

```bash
cp .env.example .env
```

| Variable | Descripción |
|----------|-------------|
| `APP_ENV` | `development` o `production` |
| `MODEL_PATH` | Ruta al modelo ResNet18 (`resnet18_best.pth`) |
| `LABEL_ENCODER_PATH` | Mapeo índice → etiqueta |
| `DATA_DIR` | Dataset fusionado (`data/fused`) |
| `METADATA_PATH` | CSV con metadata + features |
| `API_URL` | URL de la API (Streamlit) |
| `ROBOFLOW_API_KEY` | Clave Roboflow |

Configuración centralizada: `src/settings/config.py`.

## Pipeline offline

```bash
python scripts/download_dataset.py   # DS1 + DS2 → data/raw/
python scripts/fuse_datasets.py      # → data/fused/ + metadata CSV
python scripts/extract_features.py   # features tabulares en metadata
python -m src.core.train
python -m src.core.evaluate
python scripts/batch_predict.py data/fused/test  # predicción batch opcional
```

O en un solo comando:

```bash
python scripts/run_pipeline.py
```

Flujo: **download → fuse → features → train → evaluate**

### EDA

```bash
jupyter notebook notebooks/EDA_Caidas.ipynb
```

## Desarrollo local

```bash
pip install -r requirements-dev.txt
uvicorn src.api.app:app --app-dir . --reload --port 8080
streamlit run app/streamlit_app.py
```

## Tests

```bash
pytest
```

## Docker

```bash
docker compose up --build
docker compose --profile full up --build   # API + Streamlit
```

API: `http://localhost:8080`

## Buenas prácticas ML aplicadas

- Balance de clases: `WeightedRandomSampler` + pesos en `CrossEntropyLoss`
- Calidad de datos: EDA de splits, fuentes DS1/DS2 y features tabulares
- Data leakage: chequeo por filename y nombre base train/test
- Generalización: evaluación principal en `valid` si `test` es pequeño
- Separación entrenamiento (offline) vs inferencia (API Docker)
- Checkpoint del mejor modelo por `val_acc` → `resnet18_best.pth`