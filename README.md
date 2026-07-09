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
│   │   ├── labeled/                           # Imágenes ya etiquetadas ("Listo, siguiente" en label_tool.py)
│   │   ├── duplicate_review/                  # Comparaciones generadas por find_inconsistent_duplicates.py
│   │   └── bbox_log.csv                       # Una fila por PERSONA: filename, label, box normalizado, timestamp
│   ├── processed/                             # Dataset listo para entrenamiento (recortado por persona)
│   │   ├── train/
│   │   │   ├── fall/
│   │   │   └── no_fall/
│   │   ├── valid/
│   │   │   ├── fall/
│   │   │   └── no_fall/
│   │   ├── test/
│   │   │   ├── fall/
│   │   │   └── no_fall/
│   │   └── dataset_labels.csv                 # Trazabilidad: filename, source_image, label, source, split, timestamp
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
│   ├── label_tool.py                          # Etiquetador manual (servidor HTTP, multi-persona por imagen)
│   ├── find_inconsistent_duplicates.py        # Detecta labels en conflicto entre imágenes casi-duplicadas
│   ├── convert_dataset.py                     # bbox_log.csv + labeled/ → recortes en train/valid/test + CSV
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
│   │   ├── classification.py                  # Clase ImageClassifier: detecta persona, recorta y clasifica
│   │   ├── detector.py                        # PersonDetector (ssdlite320_mobilenet_v3_large, COCO)
│   │   ├── gradcam.py                         # Implementación GradCAM con hooks
│   │   └── preprocessing/
│   │       ├── transforms.py                  # Transforms de train (augmentation) y test
│   │       └── cropping.py                    # crop_to_box() — recorte con margen (offline y online)
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
│       ├── video_detection.py                 # detect_falls_from_video() — frame a frame
│       └── duplicates.py                      # dhash() + build_duplicate_groups() — perceptual hash compartido
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
│   ├── fixtures/
│   │   └── person.jpg                         # Foto real usada para probar el camino "con persona detectada"
│   ├── core/
│   │   ├── test_cropping.py                   # Tests de crop_to_box()
│   │   └── test_detector.py                   # Tests de PersonDetector
│   └── api/
│       └── test_routers.py                    # Tests de /predict, /gradcam, /dashboard/stats (con y sin persona)
│
├── docs/
│   ├── endpoints.md                           # Documentación de endpoints con ejemplos curl
│   └── arquitectura.md                        # Explicación técnica de cada archivo
│
├── Dockerfile                                 # Imagen de la API FastAPI (pre-cachea pesos del detector)
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

Clasificación binaria: dada una imagen o un video, el sistema determina si **cada persona detectada** está **caída** (`fall`) o **no caída** (`no_fall`). Una misma imagen puede tener varias personas, cada una con su propio resultado.

El dataset se construye combinando dos fuentes en un mismo pool sin etiqueta, y etiquetando **manualmente, por persona**. El etiquetado lo hace un humano mirando cada imagen y dibujando un bounding box por cada persona, no la query de búsqueda ni la predicción de un modelo, evitando el ruido del weak labeling.

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
                 label_tool.py          (por cada PERSONA: box + fall/no_fall)
                       │                 bbox_log.csv (filename, label, box)
                       ▼
              data/raw/labeled/         (imagen ya completa: "Listo, siguiente")
                       │
              convert_dataset.py        (recorta cada persona a su box + margen)
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
          train/     valid/     test/
     dataset_labels.csv  (filename, source_image, label, source, split, timestamp)
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
- **Clase asignada = lo que se ve en el frame, por persona**: `fall` si esa persona está caída/en el piso en ese instante; `no_fall` en cualquier otra postura (parada, sentada, caminando, agachada, etc.). Una misma imagen con varias personas puede tener clases distintas para cada una.
- **Ambiguas → se descartan, no se fuerzan**: si la imagen no permite decidir con confianza (mala calidad, postura ambigua, persona parcialmente fuera de cuadro), se usa `S` (saltar) o `D` (borrar) en `label_tool.py` en lugar de adivinar la clase.
- **Mismo criterio sin importar la fuente**: las imágenes de Pexels y los frames de video se mezclan en un único pool sin etiqueta (`data/raw/pool/`) antes de etiquetar, así que el origen de la imagen no influye en la decisión.
- **Trazabilidad**: cada persona confirmada (F/N) queda registrada en `data/raw/bbox_log.csv` (label + box), y al dar "Listo" la imagen se mueve a `data/raw/labeled/`. `convert_dataset.py` deja registrado en `dataset_labels.csv` la clase, el split y la procedencia real (`pool_log.csv`) de cada recorte.

### División del dataset

| Split  | Proporción |
|--------|------------|
| train  | 70%        |
| valid  | 15%        |
| test   | 15%        |

División estratificada por **clase y por fuente** (`fall`/`no_fall` × `pexels`/`video`) con semilla fija (`42`) para reproducibilidad: cada combinación se divide 70/15/15 por separado y luego se combinan los splits.

**Anti-fuga entre splits**: antes de dividir, las imágenes casi-duplicadas (frames de video separados por poco tiempo, o fotos casi idénticas) se agrupan con un perceptual hash (dHash, distancia de Hamming ≤ 4/64) y el grupo entero va al mismo split — nunca la mitad a train y la mitad a test. El EDA (`notebooks/eda.ipynb`) detectó que, sin este agrupado, un porcentaje alto de pares casi-duplicados terminaba repartido entre splits distintos, inflando artificialmente la accuracy de test.

---

## Modelos

| Modelo                   | Descripción                                            |
|--------------------------|--------------------------------------------------------|
| `resnet18.pth`           | ResNet18 con fine-tuning completo desde pesos ImageNet |
| `resnet18_quantized.pth` | Versión cuantizada (dynamic quantization int8)         |

### Detector de personas (previo a la clasificación)

Tanto el dataset de entrenamiento como la API clasifican un **recorte de la persona**, no la imagen completa. En el dataset, ese recorte sale del bounding box dibujado a mano en `label_tool.py`; en producción no hay un humano dibujando el box, así que `src/core/detector.py` (`PersonDetector`, `ssdlite320_mobilenet_v3_large` preentrenado en COCO, sin fine-tuning) detecta la persona automáticamente y aplica el mismo recorte (con el mismo margen de ~15%, ver `src/core/preprocessing/cropping.py`) antes de pasarle la imagen al clasificador. Esto evita reintroducir *training-serving skew* entre el dataset y la API.

Se eligió el detector más liviano de `torchvision` porque el proyecto corre inferencia en CPU (EC2 sin GPU): prioriza latencia sobre precisión de localización. Si no se detecta ninguna persona, `/predict`, `/predict/batch` y `/predict/video` devuelven `person_detected: false` y `label: null` en vez de forzar una clasificación; `/gradcam` devuelve un error explícito, porque no hay nada que explicar.

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
      label_tool.py           (por persona: box + fall/no_fall; una imagen
                                 puede tener varias personas y labels)
      bbox_log.csv             (filename, label, box — una fila por persona)
          │
  data/raw/labeled/            (imagen completa, "Listo, siguiente")
          │
  convert_dataset.py         (recorta cada persona a su box + margen, cropping.py)
          │
  data/processed/             (ImageFolder ready, ya recortado a la persona)
          │
      train.py                (ResNet18 + class weights + best ckpt + quantization)
          │
  models/resnet18.pth
          │
      evaluate.py  ──→  metrics.json


Pipeline de producción (online)

docker-compose up
  ├─ FastAPI  :8080     (predict / gradcam / predict/video / dashboard/stats)
  │     └─ PersonDetector (ssdlite320_mobilenet_v3_large) detecta + recorta
  │        antes de clasificar, mismo margen que convert_dataset.py
  ├─ Streamlit :8501    (UI: dashboard + predicción de imágenes y video)
  ├─ PostgreSQL :5432   (tabla predictions)
  └─ Grafana  :3000     (dashboard operacional)
```

---

## Desafíos de producción resueltos

### Data Leakage
Separación estricta train/valid/test en `convert_dataset.py` con semilla fija antes de cualquier entrenamiento, estratificada por `(clase, fuente)`. Además, las imágenes casi-duplicadas (dHash, `src/utils/duplicates.py`) se agrupan antes de dividir para que un grupo entero vaya al mismo split — sin esto, frames de video muy cercanos entre sí podían quedar repartidos entre train y test (detectado por el EDA en `notebooks/eda.ipynb`).

### Training-Serving Skew
Las transformaciones están unificadas en `transforms.py`. La API usa exactamente `get_test_transforms()`, el mismo preprocesamiento del conjunto de test.

**Recorte a la persona**: esta fue la fuente de skew más interesante del proyecto, porque no es un problema que existiera desde el principio — lo introdujimos nosotros mismos al decidir recortar el dataset al bounding box de cada persona. El dataset de entrenamiento se recorta con un box dibujado a mano por el etiquetador. En producción, evidentemente, no hay un humano dibujando un box antes de cada predicción. Si no resolvíamos esto, la API iba a clasificar la imagen completa mientras el modelo fue entrenado para clasificar recortes de personas — un skew grave y automático, garantizado en el 100% de las predicciones reales.

La solución fue agregar un detector de personas automático (`src/core/detector.py`, `PersonDetector`) que corre antes de la clasificación en producción, y que usa exactamente la misma función de recorte con margen (`crop_to_box()`, `src/core/preprocessing/cropping.py`, ~15% de margen) que ya usa `convert_dataset.py` con el box dibujado a mano. No es el detector el que se comparte entre offline y online (en el dataset no hace falta detectar nada, el humano ya marcó a la persona) — lo que se comparte es la función de recorte, para que un box de origen humano y un box de origen automático produzcan el mismo tipo de imagen recortada a la entrada del clasificador. Elegimos `ssdlite320_mobilenet_v3_large` (preentrenado en COCO, sin fine-tuning) por ser el detector más liviano disponible en `torchvision`, priorizando latencia sobre precisión de localización — el proyecto corre inferencia en CPU (pensado para EC2 sin GPU), y un detector más pesado hubiera dominado la latencia total de `/predict`.

### Calidad del etiquetado
`scripts/find_inconsistent_duplicates.py` detecta imágenes casi-duplicadas con labels en conflicto (`fall` y `no_fall` sobre la misma persona) — señal de doble confirmación accidental o de un error real de etiquetado — antes de armar el dataset final.

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

```bash
python scripts/label_tool.py
```
Abre el navegador en `http://localhost:8765`. La unidad de etiquetado es la **persona**, no la imagen: una misma foto puede tener varias personas, cada una con su propio estado.

1. Dibujá un rectángulo alrededor de una persona (click y arrastrar sobre la imagen).
2. Marcá `F`=fall o `N`=no_fall para **esa persona**. Queda registrada en `data/raw/bbox_log.csv` (una fila por persona: filename, label, box normalizado 0-1) y se muestra superpuesta en la imagen, con un botón para borrarla si te equivocaste.
3. Repetí 1-2 si hay más personas en la misma imagen.
4. `Enter` = **Listo, siguiente imagen** (requiere al menos una persona marcada): mueve la imagen de `data/raw/pool/` a `data/raw/labeled/`. `S`=saltar (no registra nada, sigue en el pool) y `D`=borrar (descarta la imagen entera) no requieren ninguna persona marcada.

Esta es la **única** vía de entrada soportada: toda imagen etiquetada tiene que haber pasado antes por `data/raw/pool/` (vía `scrape_dataset.py` o `extract_video_frames.py`), por lo que siempre queda registrada en `pool_log.csv` con su `source` real (`pexels` o `video`). No se admite copiar imágenes directamente a una carpeta de clase: `convert_dataset.py` rechaza el dataset si encuentra una fila de `bbox_log.csv` sin procedencia conocida o cuya imagen no llegó a `data/raw/labeled/`.

### 5.1 (Opcional) Revisar boxes duplicados o con labels en conflicto

```bash
python scripts/find_inconsistent_duplicates.py
```

Agrupa imágenes casi-duplicadas (mismo dHash que usa `convert_dataset.py`) y avisa si dentro de un grupo aparecen tanto `fall` como `no_fall` — señal de un posible error de etiquetado o de que la misma persona quedó confirmada dos veces. Guarda una comparación lado a lado por cada caso en `data/raw/duplicate_review/` para revisión visual. No corrige nada automáticamente: cada caso se decide a mano.

### 6. Convertir y dividir el dataset

```bash
python scripts/convert_dataset.py
```

Lee `data/raw/bbox_log.csv` (una fila por persona) y las imágenes desde `data/raw/labeled/`. Por cada persona, **recorta la imagen a su bounding box** (con ~15% de margen, ver `src/core/preprocessing/cropping.py`) y la guarda como un ejemplo de entrenamiento independiente — una misma imagen con dos personas genera dos ejemplos, potencialmente uno `fall` y otro `no_fall`. Divide en train/valid/test (70/15/15) y genera `data/processed/dataset_labels.csv` con la procedencia real de cada ejemplo.

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
| Accuracy  | 75.76% |
| Precision | 75.76% |
| Recall    | 100%   |
| F1 Score  | 86.21% |

Matriz de confusión (fall=0, no_fall=1), del `metrics.json` actualmente en el repo:

| Real / Pred. | Pred. fall | Pred. no_fall |
| --- | --- | --- |
| Real fall | 0 | 8 |
| Real no_fall | 0 | 25 |

**Estas métricas corresponden al modelo entrenado con el pipeline anterior (imagen completa, sin recorte por persona) y ya están obsoletas.** La matriz de confusión muestra que ese modelo nunca predice `fall` (0 aciertos y 0 predicciones en esa clase) — clasifica todo como `no_fall`, un modelo degenerado que solo "acierta" porque `no_fall` es la clase mayoritaria. Las métricas de precision/recall/accuracy de la tabla son las de `no_fall` (default de scikit-learn); el recall real de `fall` es **0%**.

Con el nuevo pipeline de este branch (recorte por persona vía `PersonDetector`/`crop_to_box`, split anti-fuga por dHash, EDA con verificación de calidad), hace falta reconstruir el dataset (`convert_dataset.py`) y reentrenar (`python -m src.core.train` + `python -m src.core.evaluate`) para tener métricas representativas del pipeline actual — no se incluyen números "de mentira" hasta no correr ese entrenamiento real.
