obligatorio/
│
├── data/                         # Dataset (NO subir a GitHub)
│   ├── raw/                      # dataset original de Roboflow
│   ├── processed/                # dataset convertido a clasificación
│   │   ├── train/
│   │   │   ├── acostado/
│   │   │   └── no_acostado/
│   │   ├── valid/
│   │   └── test/
│
├── models/                       # Modelos entrenados
│   ├── resnet18.pth
    ├──resnet18_quantized.pth     # cunando queremos usarlo?
│
├── notebooks/                    # EDA y experimentación
│   └── eda.ipynb                 # falta hacerlo
│
├── scripts/                      # Scripts auxiliares
│   ├── download_dataset.py       # descarga Roboflow
│   ├── convert_dataset.py        # convierte a clasificación
│
├── src/                          # Código fuente principal
│
│   ├── api/                      # API (FastAPI) Se utilizó FastAPI, que permite generar documentación automática de la API mediante Swagger, facilitando la exploración y prueba de los endpoints.
│   │   ├── routers/
│   │   │   ├── predict.py        # endpoint CNN
│   │   │   └── predict_yolo.py   # endpoint YOLO
│   │   └── app.py                # integración FastAPI
│
│   ├── core/                     # Lógica ML
│   │   ├── model.py              # ResNet18
│   │   ├── train.py              # revisar mejorar,cambiar la cantidad de epochs
│   │   ├── evaluate.py           # Las métricas del modelo se calculan offline utilizando un conjunto de test, ya  que en producción no se dispone de etiquetas reales para comparar las predicciones
│   │   ├── yolo_model.py         # baseline YOLO
│   │   ├── gradcam.py            # explicabilidad
│   │
│   ├── preprocessing/            # Transformaciones
│   │   └── transforms.py         # falta mejorarlo
│
│   ├── data/                     # Carga de datos
│   │   └── dataset.py
│
│   ├── utils/                    # Utilidades
│   │   ├── metrics.py
│   │   └── logger.py             # registro de eventos,te permite debuggear mejor
│
│   ├── settings/                 # Configuración
│   │   └── config.py             # aca esta el modelo a utilizar y OS
│
├── app/                          # UI (Streamlit)
│   └── streamlit_app.py          # este es el frontend (permite subir mas de una imagen)
│
├── tests/                        # Tests
│   └── test_api.py               # esto hay que adecuarlo
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore



# Hospital Bed Detector

## Problema
Clasificación binaria para detectar si un paciente está acostado.

## Modelos
- CNN ResNet18 (modelo de clasificacion)
- YOLO baseline (modelo de deteccion)

## API
- /predict
- /predict_yolo
- /gradcam

## Run en local

pip install -r requirements.txt
python src/core/train.py
uvicorn src.api.app:app --reload

# tengo otro venv
deactivate
Rename-Item venv venv_old
python -m venv venv
.\venv\Scripts\Activate.ps1


# Pasos

# offline 
1. pip install -r requirements.txt
2. python scripts/download_dataset.py
3. python scripts/convert_dataset.py
4. python -m src.core.train

download → convert → train → evaluate

# Produccion
# API (FastAPI)
5. docker-compose up --build 
# front
6. streamlit (dockerfile)

# Para detener el docker docker-compose down -v


# TEST
7. python -m src.core.evaluate
8. python scripts/compare_models.py



El sistema se compone de:

un pipeline offline de entrenamiento,
una API para inferencia en producción,
un script para evaluación comparativa,
y una interfaz de usuario con Streamlit para interacción con el modelo.

# En sistemas de ML en producción, el entrenamiento y la inferencia suelen separarse. El entrenamiento es costoso y se ejecuta offline, mientras que la API en producción debe ser ligera, rápida y estable.




train.py → crea modelo ✅
model.py → define modelo ✅
gradcam.py → explica ✅
API → usa modelo ✅
Streamlit → usa API ✅



                    Pipeline completo
                ┌───────────────────────┐
                │   ROBFLOW DATASET     │
                │ (imágenes + labels)   │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │ download_dataset.py   │
                │ (descarga dataset)    │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │   data/raw            │
                │ (formato YOLO)        │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │ convert_dataset.py    │
                │ (detección → clases)  │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │ data/processed        │
                │ (ImageFolder ready)   │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │    train.py           │
                │ (entrena ResNet18)    │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │ models/resnet18.pth   │
                │ (modelo final)        │
                └──────────┬────────────┘

Produccion 

                ┌───────────────────────┐
                │ docker-compose up     │
                │ (levanta API y front) │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │    FastAPI API        │
                │  (src/api/app.py)     │
                └──────────┬────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  /predict     │  │ /predict_yolo │  │  /gradcam     │
│  CNN model    │  │ YOLO model    │  │ GradCAM       │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        ▼                  ▼                  ▼
   model.py         YOLO model         gradcam.py
   (ResNet18)       baseline           explicación

Visualizacion
                ┌───────────────────────┐
                │   Streamlit App       │
                │  (src/app/...)        │
                └──────────┬────────────┘
                           │
                           ▼
                (envía imagen a API)
                           │
                           ▼
          ┌──────────────────────────────────┐
          │       RESULTADOS MOSTRADOS       │
          │                                  │
          │ ✅ CNN predicción                │
          │ ✅ YOLO predicción               │
          │ ✅ GradCAM (heatmap)             │
          └──────────────────────────────────┘

# Notas:
# Data lakage
Se evitó data leakage mediante la separación estricta de los conjuntos de entrenamiento y prueba, y utilizando transformaciones distintas para cada uno.
NO mezclar transforms

# TRAINING-SERVING SKEW
 ocurre cuando:
modelo ve datos distintos en producción que en training

Para prevenir training-serving skew, se unificaron las transformaciones utilizadas en entrenamiento y en producción mediante un módulo compartido de preprocessing.


train.py Entrenar modelo 
Se utilizó CrossEntropyLoss ya que es la función estándar para problemas de clasificación, permitiendo medir la diferencia entre las predicciones del modelo y las etiquetas reales.
Se utilizó Adam debido a su capacidad de adaptar automáticamente el learning rate y acelerar la convergencia durante el entrenamiento.

model.py Se utilizó un modelo ResNet18 preentrenado sobre el cual se aplicó fine-tuning, entrenando todas las capas del modelo para adaptarlo al problema específico.