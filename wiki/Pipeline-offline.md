# Pipeline offline

[[← Inicio|Home]] · [[Modelo y entrenamiento]]

---

## Flujo completo

```
1. download_dataset.py   →  data/raw/
2. fuse_datasets.py      →  data/fused/ + metadata CSV
3. extract_features.py   →  CSV enriquecido con features
4. src.core.train        →  models/resnet18_best.pth
5. src.core.evaluate     →  métricas en consola
```

---

## Comando único

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

### Flags útiles

| Flag | Efecto |
|------|--------|
| `--skip-download` | No descarga de Roboflow (usa `data/raw/` existente) |
| `--skip-train` | Solo prepara datos, no entrena |

---

## Paso a paso

### 1. Descarga (`scripts/download_dataset.py`)

Baja DS1 y DS2 desde Roboflow a `data/raw/`.

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py
```

Requiere `ROBOFLOW_API_KEY` en `.env`.

---

### 2. Fusión (`scripts/fuse_datasets.py`)

Unifica ambos datasets en `data/fused/`:

```
data/fused/
  train/fall/       train/not_fall/
  valid/fall/       valid/not_fall/
  test/fall/        test/not_fall/
```

Genera: `data/metadata/fall_dataset_fused_metadata.csv`

---

### 3. Features (`scripts/extract_features.py`)

Calcula 9 features tabulares por imagen (brillo, contraste, etc.) y las agrega al CSV.

> El entrenamiento CNN usa solo imágenes; las features sirven para EDA e informe.

---

### 4. Entrenamiento

Ver [[Modelo y entrenamiento]].

```powershell
.\.venv\Scripts\python.exe -m src.core.train
```

---

### 5. Evaluación

```powershell
.\.venv\Scripts\python.exe -m src.core.evaluate
```

---

## Predicción batch (opcional)

```powershell
.\.venv\Scripts\python.exe scripts\batch_predict.py data\fused\test
```

Salida: `data/metadata/batch_predictions.csv`

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```
