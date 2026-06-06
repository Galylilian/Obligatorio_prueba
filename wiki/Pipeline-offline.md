# Pipeline offline

[[← Inicio|Home]] · [[Modelo y entrenamiento]]

Todo lo que pasa **antes** de levantar la API: bajar datos, entrenar y evaluar. Corré esto una vez (o cuando quieras re-entrenar).

---

## El camino completo

```
1. download   →  baja fotos de Roboflow
2. fuse       →  une DS1 + DS2 en una sola carpeta
3. features   →  calcula números extra por imagen (para el EDA)
4. train      →  entrena ResNet18
5. evaluate   →  te dice qué tan bien le fue
```

---

## La forma fácil: un solo comando

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

Se sienta, tomá un café ☕ — en CPU puede tardar 20–40 minutos.

**Atajos:**
- `--skip-download` → ya tenés `data/raw/`, no vuelve a bajar
- `--skip-train` → solo prepara datos, sin entrenar

---

## Paso a paso (si preferís ir de a uno)

### 1. Descargar (`download_dataset.py`)

Trae dos datasets de Roboflow a `data/raw/`.

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py
```

Necesitás `ROBOFLOW_API_KEY` en `.env`.

---

### 2. Fusionar (`fuse_datasets.py`)

Mezcla todo en `data/fused/` con carpetas claras:

```
train/fall/    train/not_fall/
valid/fall/    valid/not_fall/
test/fall/     test/not_fall/
```

También genera un CSV con metadata en `data/metadata/`.

---

### 3. Features (`extract_features.py`)

Por cada foto calcula brillo, contraste, etc. **El CNN no usa esto para entrenar** — es para el notebook EDA y el informe.

---

### 4. Entrenar

```powershell
.\.venv\Scripts\python.exe -m src.core.train
```

Detalles en [[Modelo y entrenamiento]].

---

### 5. Evaluar

```powershell
.\.venv\Scripts\python.exe -m src.core.evaluate
```

Te imprime accuracy, precision, recall y matriz de confusión.

---

## Extras

**Predicción en lote** (muchas fotos de una carpeta):
```powershell
.\.venv\Scripts\python.exe scripts\batch_predict.py data\fused\test
```

**Tests automáticos:**
```powershell
.\.venv\Scripts\python.exe -m pytest
```

---

Listo el pipeline → seguí con [[API e inferencia]] o [[Streamlit]].
