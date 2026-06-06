# API e inferencia

[[← Inicio|Home]] · [[Streamlit]]

---

## Levantar la API

```powershell
cd Obligatorio_prueba
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --app-dir . --host 127.0.0.1 --port 8080 --reload
```

Documentación interactiva: **http://localhost:8080/docs**

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Estado del servicio |
| GET | `/` | Lista de endpoints |
| POST | `/predict` | Clasificación Fall / Not Fall |
| POST | `/gradcam` | Mapa de calor explicativo (JPEG) |

---

## POST `/predict`

**Request:** `multipart/form-data` con campo `file` (imagen JPG/PNG)

**Response:**

```json
{
  "prediction": 0,
  "label": "fall",
  "confidence": 0.9542,
  "probabilities": {
    "fall": 0.9542,
    "not_fall": 0.0458
  }
}
```

- `confidence`: probabilidad softmax de la clase predicha (0–1)
- Confianza **< 0.70** → el modelo no está seguro

### Ejemplo curl

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@data/fused/test/fall/ejemplo.jpg"
```

---

## POST `/gradcam`

Devuelve imagen JPEG con heatmap superpuesto.

```bash
curl -X POST http://localhost:8080/gradcam \
  -F "file=@imagen.jpg" \
  --output gradcam.jpg
```

**Colores:** rojo = zonas más influyentes en la decisión; azul = poco relevante.

---

## Reiniciar tras reentrenar

La API cachea el modelo en memoria. Después de `train.py`, **reiniciar uvicorn** para cargar los nuevos pesos.

---

## Arquitectura inferencia

```
Imagen → transform (224×224) → ResNet18 → softmax → label + confidence
```

Código: `src/api/routers/predict.py` · `src/core/inference.py`
