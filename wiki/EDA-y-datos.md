# EDA y datos

[[← Inicio|Home]] · [[Pipeline offline]]

---

## Estructura de datos

| Carpeta | Contenido |
|---------|-----------|
| `data/raw/` | DS1 + DS2 descargados de Roboflow |
| `data/fused/` | Dataset unificado (train/valid/test × fall/not_fall) |
| `data/metadata/` | CSV con paths, labels, features tabulares |

**No se versionan en Git** — se generan con el pipeline.

---

## Balance típico (~1.373 imágenes)

| Split | fall | not_fall |
|-------|------|----------|
| train | ~550 | ~514 |
| valid | ~100 | ~77 |
| test | ~73 | ~59 |

---

## Notebook EDA

Archivo: `notebooks/EDA_Caidas.ipynb`

### Modo local (recomendado)

Si ya corriste el pipeline:

1. Abrir notebook en Cursor/VS Code
2. Kernel: **Python (obligatorio)**
3. Ejecutar celda **Modo local**
4. **Saltear** celdas de descarga/fusión Roboflow
5. Ejecutar balance, leakage, features, UMAP

### Qué analiza

- Balance por split y clase
- Contribución DS1 vs DS2
- Propiedades visuales (brillo, contraste)
- **Data leakage** (duplicados entre train/test)
- UMAP de features tabulares

---

## Features tabulares

9 valores numéricos por imagen (brillo, contraste, aspect ratio, etc.).

Generados por: `scripts/extract_features.py`  
Código: `src/features/tabular_features.py`

Usados en EDA; el CNN entrena solo con píxeles.

---

## Data leakage

Chequeos en `src/preprocessing/eda_stats.py`:

- Mismo filename base en train y test
- Duplicados exactos entre splits

Resultado esperado tras fusión correcta: **0 casos** de leakage.

---

## Prefijos de archivos

| Prefijo | Origen |
|---------|--------|
| `DS1_` | fall-detection-raskl |
| `DS2_` | fa-nunl5 |
