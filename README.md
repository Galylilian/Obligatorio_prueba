# Fall Detector

Sistema de Machine Learning para clasificación binaria de caídas en imágenes y video. Detecta si una persona está caída o no utilizando una ResNet18 con fine-tuning, expuesta mediante una API REST con FastAPI y una interfaz visual con Streamlit.

---

## Contenido del proyecto

```
Obligatorio_prueba/
│
├── data/                                      # Dataset (NO subir a GitHub)
│   ├── raw/                                   # Imágenes sin procesar
│   │   ├── pool/                              # scrape_dataset.py + extract_video_frames.py (sin etiqueta)
│   │   │   └── pool_log.csv                   # Log: filename, source (pexels/video), timestamp, ...
│   │   ├── fall/                              # Imágenes etiquetadas como caída (manual)
│   │   └── no_fall/                           # Imágenes etiquetadas como no caída (manual)
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
│   │   └── dataset_labels.csv                 # Trazabilidad: imagen, label, split, timestamp
│   └── video/                                 # Videos subidos y frames con caídas
│       ├── input/                             # Videos propios para extraer frames (dataset)
│       ├── uploads/                           # Videos subidos vía API (inferencia)
│       └── frames/                            # Frames marcados como "fall" por /predict/video
│
├── models/                                    # Modelos entrenados (NO subir a GitHub)
│   ├── resnet18.pth                           # Modelo principal (float32)
│   └── resnet18_quantized.pth                 # Modelo cuantizado (int8, menor latencia)
│
├── notebooks/
│   └── eda.ipynb                              # Análisis exploratorio del dataset
│
├── queries/
│   └── people.txt                             # Queries genéricas para el scraper (sin etiqueta)
│
├── scripts/
│   ├── scrape_dataset.py                      # Scraper Pexels → data/raw/pool/
│   ├── extract_video_frames.py                # Frames de data/video/input/ → data/raw/pool/
│   ├── label_tool.py                          # Etiquetador manual (servidor HTTP)
│   ├── convert_dataset.py                     # fall/ + no_fall/ → train/valid/test + CSV
│   ├── compare_models.py                      # Compara predicciones entre modelos
│   ├── benchmark_quantization.py               # Latencia normal vs. cuantizado sobre test_loader real
│   └── video_predict.py                       # Corre detect_falls_from_video() sin la API
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
│   │   ├── train.py                           # Entrenamiento con class weights + quantization
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
│   │   ├── database.py                        # Conexión SQLAlchemy + get_db() + init_db()
│   │   └── models.py                          # Tabla predictions en PostgreSQL
│   │
│   ├── settings/
│   │   └── config.py                          # MODEL_TYPE, MODEL_PATH, DEVICE, DATABASE_URL
│   │
│   └── utils/
│       ├── metrics.py                         # compute_metrics() — accuracy, F1, precision, recall
│       ├── logger.py                          # get_logger() — logger centralizado
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
│           ├── dashboard.yml                  # Proveedor de dashboards
│           └── fall_detector.json             # Dashboard pre-configurado con 7 paneles
│
├── tests/
│   ├── conftest.py                            # Fixtures — cliente de test con DB en memoria
│   ├── test_api.py
│   └── api/
│       └── test_routers.py                    # Tests de /predict, /gradcam, /dashboard/stats
│
├── docs/
│   ├── endpoints.md                           # Documentación de endpoints con ejemplos curl
│   └── arquitectura.md                        # Explicación técnica de cada archivo
│
├── Dockerfile                                 # Imagen de la API FastAPI
├── Dockerfile.streamlit                       # Imagen de Streamlit
├── docker-compose.yml                         # Orquestación: FastAPI + Streamlit + PostgreSQL + Grafana
├── requirements.txt                           # Dependencias de producción
├── metrics.json                               # Métricas del modelo (generado por evaluate.py)
├── benchmark_quantization.json                 # Latencia normal vs. cuantizado (generado por benchmark_quantization.py)
├── .env                                       # Variables de entorno (NO subir a GitHub)
├── .gitignore
└── README.md
```

---

## Problema

Clasificación binaria: dada una imagen o un video, el sistema determina si la persona está **caída** (`fall`) o **no caída** (`no_fall`).

El dataset se construye combinando dos fuentes en un mismo pool sin etiqueta, y etiquetando **manual** imagen por imagen. El etiquetado lo hace un humano mirando cada imagen, no la query de búsqueda ni la predicción de un modelo, evitando el ruido del weak labeling.

---

## Dataset

| Fuente        | Cómo se suma al pool                                                 | Método de etiquetado     | Clases             |
|---------------|-----------------------------------------------------------------------|--------------------------|--------------------|
| Pexels (API)  | `scrape_dataset.py`                                                    | Etiquetado manual humano | `fall` / `no_fall` |
| Video propio  | `extract_video_frames.py` (todos los frames, sin filtro del modelo)   | Etiquetado manual humano | `fall` / `no_fall` |

`data/raw/pool/pool_log.csv` guarda la procedencia real de cada imagen (`source=pexels` o `source=video`), y `convert_dataset.py` la propaga a `data/processed/dataset_labels.csv` para mantener trazabilidad completa de dónde salió cada imagen del dataset final.

### Pipeline de construcción

```text
queries/people.txt                    data/video/input/*.mp4
       │                                      │
       ▼                                      ▼
scrape_dataset.py               extract_video_frames.py
       │                                      │
       └──────────────┬───────────────────────┘
                       ▼
              data/raw/pool/            (Pexels + frames de video, sin etiqueta)
              pool_log.csv              (procedencia: pexels / video)
                       │
                 label_tool.py          (etiquetado manual en el navegador)
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
     data/raw/fall/        data/raw/no_fall/
            │                     │
            └──────────┬──────────┘
                       ▼
              convert_dataset.py       (usa pool_log.csv para el source real)
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
          train/     valid/     test/
     dataset_labels.csv  (filename, label, source, split, timestamp)
```

### Queries utilizadas

Las queries en `queries/people.txt` son genéricas (mezcla de candidatos a ambas clases). La etiqueta real la asigna el humano al mirar la imagen, no el término de búsqueda.

```text
person standing indoors
person walking street
elderly person walking cane
person jogging park
cctv people corridor
emergency responder fallen patient
man collapsed office floor
person lying on ground
...
```

### Política de etiquetado

- **Un único criterio, humano y visual**: se etiqueta mirando la imagen, no la query que la trajo al pool ni ninguna predicción de un modelo. Esto evita el ruido típico del *weak labeling* (asumir la clase a partir del texto de búsqueda).
- **Clase asignada = lo que se ve en el frame**: `fall` si la persona está caída/en el piso en ese instante; `no_fall` en cualquier otra postura (parada, sentada, caminando, agachada, etc.).
- **Ambiguas → se descartan, no se fuerzan**: si la imagen no permite decidir con confianza (mala calidad, postura ambigua, persona parcialmente fuera de cuadro), se usa `S` (saltar) o `D` (borrar) en `label_tool.py` en lugar de adivinar la clase.
- **Mismo criterio sin importar la fuente**: las imágenes de Pexels y los frames de video se mezclan en un único pool sin etiqueta (`data/raw/pool/`) antes de etiquetar, así que el origen de la imagen no influye en la decisión.
- **Trazabilidad**: cada decisión de etiquetado mueve el archivo a `data/raw/fall/` o `data/raw/no_fall/`, y `convert_dataset.py` deja registrado en `dataset_labels.csv` la clase, el split y la procedencia real (`pool_log.csv`) de cada imagen.

### División del dataset

| Split  | Proporción |
|--------|------------|
| train  | 70%        |
| valid  | 15%        |
| test   | 15%        |

División estratificada por clase con semilla fija (`42`) para reproducibilidad.

---

## Modelos

| Modelo                   | Descripción                                            |
|--------------------------|--------------------------------------------------------|
| `resnet18.pth`           | ResNet18 con fine-tuning completo desde pesos ImageNet |
| `resnet18_quantized.pth` | Versión cuantizada (dynamic quantization int8)         |

### Desbalance de clases

El entrenamiento usa **class weights** inversamente proporcionales a la frecuencia de cada clase:

```text
weight[clase] = n_samples / (n_clases × count[clase])
```

Esto penaliza más los errores en la clase minoritaria (`fall`), que es la clase crítica del problema.

### Selección de modelo en producción

Se controla con `MODEL_TYPE` en `docker-compose.yml`:

```yaml
- MODEL_TYPE=normal      # usa resnet18.pth (default)
- MODEL_TYPE=quantized   # usa resnet18_quantized.pth
```

### Impacto de la quantization en latencia

`scripts/benchmark_quantization.py` mide la latencia de inferencia en CPU (batch_size=1, igual que `/predict`) sobre el `test_loader` real, comparando `resnet18.pth` contra `resnet18_quantized.pth`. Resultado (`benchmark_quantization.json`):

| Variante                          | Latencia media | Tamaño en disco | Speedup |
|-----------------------------------|-----------------|------------------|---------|
| Normal (fp32)                     | ~25-34 ms       | 42.71 MB         | 1.00x   |
| Cuantizado (int8, solo Linear)     | ~27-36 ms       | 42.71 MB         | ~0.96x  |
| Cuantizado (int8, Linear+Conv2d)   | ~29-33 ms       | 42.71 MB         | ~0.89x  |

**Conclusión: en este modelo, la quantization dinámica no reduce la latencia ni el tamaño en disco.** `torch.ao.quantization.quantize_dynamic` solo cuantiza capas `Linear` (y `LSTM`/`GRU`/`RNN`, no aplicables acá). En ResNet18 la única `Linear` es la `fc` final (512→2), una capa minúscula frente a los ~44 MB de capas `Conv2d` del backbone, que es donde realmente está el costo computacional. Al no poder cuantizar las convoluciones (`Conv2d` no está soportado por la cuantización dinámica de PyTorch en esta versión — se verificó empíricamente incluyéndolo en el set de módulos a cuantizar y las capas quedan sin tocar), el beneficio esperado no se materializa; incluso se observa un leve overhead adicional.

La alternativa real para impactar la latencia sería **cuantización estática (PTQ)** sobre las capas convolucionales (fusión Conv+BN+ReLU, calibración con datos reales), que queda fuera del alcance actual del proyecto.

---

## Arquitectura del sistema

```text
Pipeline de entrenamiento (offline)

scrape_dataset.py          extract_video_frames.py
  └───────┬──────────────────────┘
          ▼
  data/raw/pool/             (Pexels + frames de video, sin etiqueta)
  pool_log.csv                (procedencia: pexels / video)
          │
      label_tool.py
          │
  ┌───────┴───────┐
fall/          no_fall/
  └───────┬───────┘
          │
  convert_dataset.py
          │
  data/processed/             (ImageFolder ready)
          │
      train.py                (ResNet18 + class weights + best ckpt + quantization)
          │
  models/resnet18.pth
          │
      evaluate.py  ──→  metrics.json


Pipeline de producción (online)

docker-compose up
  ├─ FastAPI  :8080     (predict / gradcam / predict/video / dashboard/stats)
  ├─ Streamlit :8501    (UI: dashboard + predicción de imágenes y video)
  ├─ PostgreSQL :5432   (tabla predictions)
  └─ Grafana  :3000     (dashboard operacional)
```

---

## Desafíos de producción resueltos

### Data Leakage
Separación estricta train/valid/test en `convert_dataset.py` con semilla fija antes de cualquier entrenamiento.

### Training-Serving Skew
Las transformaciones están unificadas en `transforms.py`. La API usa exactamente `get_test_transforms()`, el mismo preprocesamiento del conjunto de test.

### Desbalance de clases
`CrossEntropyLoss` recibe pesos calculados automáticamente desde la distribución real del conjunto de train, sin necesidad de configuración manual.

---

## Requerimientos electivos implementados

| Electivo | Implementación |
| --- | --- |
| Scraper de datos | `scrape_dataset.py` — Pexels, etiquetado manual, log CSV |
| Explicabilidad | GradCAM sobre `layer4[-1]` de ResNet18 |
| Visualización | Streamlit con dashboard, predicción de imágenes y video |
| Optimización de modelos | Data Augmentation + Quantization dinámica (int8) + Class Weights |

---

## Cómo correr el proyecto

### 1. Configurar variables de entorno

Crear `.env` en la raíz:

```env
PEXELS_API_KEY=tu_key_aqui
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/falldetector
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

Descarga imágenes desde Pexels a `data/raw/pool/` usando las queries de `queries/people.txt`. Las imágenes no tienen etiqueta aún. Cada imagen queda registrada en `data/raw/pool/pool_log.csv` con `source=pexels`.

### 4. Sumar frames de video al pool (opcional)

```bash
python scripts/extract_video_frames.py
```

Recorre los videos de `data/video/input/` y guarda **todos** los frames (sin usar el modelo para preseleccionar) en `data/raw/pool/`, junto con las imágenes de Pexels. Cada frame queda registrado en `pool_log.csv` con `source=video`, el video de origen y el timestamp dentro del video. Usar `--every N` para guardar 1 de cada N frames en vez de todos.

### 5. Etiquetar imágenes

**Opción A — Herramienta visual:**
```bash
python scripts/label_tool.py
```
Abre el navegador en `http://localhost:8765`. Atajos: `F`=fall, `N`=no_fall, `S`=saltar, `D`=borrar.
Las imágenes etiquetadas se mueven automáticamente a `data/raw/fall/` o `data/raw/no_fall/`, sin importar si vinieron de Pexels o de un video: el pool las mezcla y el etiquetado es el mismo para todas.

**Opción B — Subida manual:**
Copiar directamente las imágenes ya clasificadas a `data/raw/fall/` y `data/raw/no_fall/`. En este caso no quedan registradas en `pool_log.csv`, y `convert_dataset.py` les asigna `source=unknown`.

### 6. Convertir y dividir el dataset

```bash
python scripts/convert_dataset.py
```

Lee desde `data/raw/fall/` y `data/raw/no_fall/`, divide en train/valid/test (70/15/15) y genera `data/processed/dataset_labels.csv` con la procedencia real de cada imagen (tomada de `pool_log.csv`).

### 7. Entrenar el modelo

```bash
python -m src.core.train
```

Entrena ResNet18 con fine-tuning desde pesos ImageNet. Calcula class weights automáticamente desde el dataset de train. Guarda el mejor checkpoint y genera el modelo cuantizado.

### 8. Evaluar el modelo

```bash
python -m src.core.evaluate
```

Genera `metrics.json` con accuracy, precision, recall, F1 y matriz de confusión sobre el conjunto de test.

### 9. Levantar en producción con Docker

```bash
docker-compose up --build
```

| Servicio | URL |
|----------|-----|
| FastAPI (Swagger) | http://localhost:8080/docs |
| Streamlit | http://localhost:8501 |
| Grafana | http://localhost:3000 (admin / admin) |

### 10. Detener los servicios

```bash
docker-compose down -v
```

### 11. Correr tests

```bash
pytest tests/
```

---

## Despliegue en AWS (EC2, sin GPU)

El proyecto está pensado para correr en una instancia EC2 de AWS Academy sin GPU, usando los mismos `Dockerfile`/`docker-compose.yml` que en local. `requirements.txt` instala PyTorch en su build CPU-only (`torch==2.7.1+cpu`), evitando descargar los paquetes con CUDA (~5 GB) que no sirven de nada sin GPU y pueden llenar el disco de la instancia.

### 1. Instancia

- EC2 (AWS Academy), tipo `t3.medium` o superior recomendado (el fine-tuning ya viene entrenado, en producción solo se hace inferencia CPU).
- Security Group con los puertos `8080` (FastAPI), `8501` (Streamlit) y `3000` (Grafana) abiertos.
- Docker y Docker Compose instalados en la instancia.

### 2. Clonar y configurar

```bash
git clone <repo>
cd Obligatorio_prueba
```

Crear el `.env` en la raíz (no se sube a GitHub):

```env
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/falldetector
MODEL_TYPE=normal
```

`PEXELS_API_KEY` solo hace falta si vas a correr `scrape_dataset.py` en la instancia; para servir el modelo ya entrenado no es necesario.

### 3. Subir los modelos entrenados

Los `.pth` en `models/` no están en git (pesan demasiado). Copiarlos a la instancia, por ejemplo con `scp`:

```bash
scp -r models/ ubuntu@<ip-ec2>:~/Obligatorio_prueba/models/
```

### 4. Levantar los servicios

```bash
docker-compose up --build -d
```

Al ser CPU-only, el build es considerablemente más liviano y rápido que con soporte CUDA.

| Servicio | URL |
| -------- | --- |
| FastAPI (Swagger) | `http://<ip-ec2>:8080/docs` |
| Streamlit | `http://<ip-ec2>:8501` |
| Grafana | `http://<ip-ec2>:3000` (admin / admin) |

### 5. Verificar

```bash
curl http://<ip-ec2>:8080/health
```

---

## Métricas del modelo

| Métrica   | Valor  |
|-----------|--------|
| Accuracy  | 88.00% |
| Precision | 94.12% |
| Recall    | 88.89% |
| F1 Score  | 91.43% |

Evaluado sobre el conjunto de test (25 imágenes) con el modelo `resnet18.pth`.

Matriz de confusión (fall=0, no_fall=1):

| Real / Pred. | Pred. fall | Pred. no_fall |
| --- | --- | --- |
| Real fall | 6 | 1 |
| Real no_fall | 2 | 16 |

La precision es alta (94%) — cuando el modelo dice "fall", casi siempre acierta. El recall del 89% implica que se pierden 2 caídas reales de 9. Para producción, ese es el error más crítico y mejora con más datos de entrenamiento en la clase `fall`.
