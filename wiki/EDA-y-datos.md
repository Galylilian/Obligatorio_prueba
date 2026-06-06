# EDA y datos

[[← Inicio|Home]] · [[Pipeline offline]]

Esta sección es para el **informe**: entender tus datos antes de confiar ciegamente en el modelo.

---

## ¿Dónde vive cada cosa?

| Carpeta | Qué hay |
|---------|---------|
| `data/raw/` | Lo que bajó Roboflow (DS1 y DS2 por separado) |
| `data/fused/` | Todo unificado, listo para entrenar |
| `data/metadata/` | CSV con info de cada foto + features |

No están en GitHub — los generás con el pipeline en tu PC.

---

## ¿Está balanceado?

Aproximadamente, con ~1.373 fotos:

| Split | fall | not_fall |
|-------|------|----------|
| train | ~550 | ~514 |
| valid | ~100 | ~77 |
| test | ~73 | ~59 |

Está razonablemente equilibrado. El notebook te lo grafica mejor.

---

## Notebook `EDA_Caidas.ipynb`

### Cómo correrlo sin volverte loco

Si **ya corriste el pipeline**, no hace falta re-descargar Roboflow:

1. Abrí el notebook en Cursor/VS Code
2. Kernel: **Python (obligatorio)**
3. Ejecutá la celda **"Modo local"**
4. **Saltá** las celdas de descarga y fusión de Roboflow
5. Seguí con balance, leakage, gráficos, UMAP

### Qué vas a mostrar en el informe

- Cuántas fotos hay por clase y por split
- Cuánto aporta DS1 vs DS2
- Brillo/contraste por clase
- Si hay **fugas de datos** (misma foto en train y test — malo)
- UMAP de features tabulares

---

## Features tabulares (¿qué son?)

Son 9 números por imagen: brillo, contraste, proporción, etc.  
El CNN **no los usa** para clasificar — sirven para analizar y enriquecer el informe.

---

## Data leakage — ojo con esto

Si la misma foto aparece en train y test, las métricas mienten (memorizó, no generalizó).

Nuestro pipeline chequea duplicados. Resultado esperado: **0 casos**. Si ves alguno, algo salió mal en la fusión.

---

## Prefijos en los nombres de archivo

- `DS1_` → viene de fall-detection-raskl
- `DS2_` → viene de fa-nunl5

Así sabés de dónde salió cada imagen.
