# API e inferencia

[[← Inicio|Home]] · [[Streamlit]]

La API es cómo el resto del mundo usa tu modelo: mandás una foto, te devuelve `fall` o `not_fall`.

---

## Levantar la API

```powershell
cd Obligatorio_prueba
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --app-dir . --port 8080 --reload
```

Dejá esa terminal abierta. Probala en: **http://localhost:8080/docs** (Swagger — podés subir fotos ahí mismo).

---

## Endpoints que importan

| Ruta | Qué hace |
|------|----------|
| `GET /health` | "¿Estás vivo?" → `{"status":"ok"}` |
| `POST /predict` | Clasifica la imagen |
| `POST /gradcam` | Devuelve foto con mapa de calor |

---

## `/predict` — la estrella

Subís una imagen, recibís algo así:

```json
{
  "label": "fall",
  "confidence": 0.9542,
  "probabilities": {
    "fall": 0.9542,
    "not_fall": 0.0458
  }
}
```

- **confidence** = qué tan seguro está (de 0 a 1)
- Si es **menor a 0.70**, tratá la respuesta con cautela

**Ejemplo con curl:**
```bash
curl -X POST http://localhost:8080/predict -F "file=@mi_foto.jpg"
```

---

## `/gradcam` — ¿dónde miró?

Devuelve una imagen JPEG con colores encima:
- **Rojo/amarillo** → zonas que más influyeron
- **Azul** → casi no importaron

No es perfecto, pero ayuda a explicar la decisión en el informe.

---

## Reentrenaste el modelo?

La API guarda el modelo en memoria. Después de `train.py`, **reiniciá uvicorn** (Ctrl+C y volvé a correr el comando) para que cargue los pesos nuevos.

---

## Por dentro (resumen)

```
Foto → resize 224×224 → ResNet18 → softmax → etiqueta + confianza
```

Código: `src/api/routers/predict.py`
