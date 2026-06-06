# Detección de caídas — Wiki del proyecto

**Repositorio:** [Obligatorio_prueba (rama `marcelo`)](https://github.com/Galylilian/Obligatorio_prueba/tree/marcelo)  
**Curso:** Machine Learning en Producción — ORT Uruguay

---

## ¿Qué hace este proyecto?

Sistema de clasificación binaria **Fall / Not Fall** a partir de imágenes:

| Clase | Significado |
|-------|-------------|
| `fall` | Persona en situación de caída |
| `not_fall` | Situación normal (de pie, sentada, etc.) |

Incluye pipeline offline de entrenamiento, API REST (FastAPI), interfaz Streamlit, explicabilidad Grad-CAM y EDA documentado.

---

## Arquitectura general

```
Roboflow (DS1 + DS2)
       ↓
  Pipeline offline  →  models/resnet18_best.pth
       ↓
  API FastAPI :8080  ←  Streamlit :8501
```

---

## Índice de la wiki

| Página | Contenido |
|--------|-----------|
| [[Instalación y configuración]] | Python, venv, `.env`, dependencias |
| [[Pipeline offline]] | Download → fuse → features → train → evaluate |
| [[Modelo y entrenamiento]] | ResNet18, fine-tuning, métricas |
| [[API e inferencia]] | Endpoints `/predict`, `/gradcam`, confianza |
| [[Streamlit]] | Interfaz web para subir imágenes |
| [[EDA y datos]] | Notebook, balance, leakage, features |
| [[Docker]] | Despliegue con contenedores |
| [[Solución de problemas]] | Errores frecuentes en Windows |

---

## Inicio rápido

```powershell
git clone -b marcelo https://github.com/Galylilian/Obligatorio_prueba.git
cd Obligatorio_prueba
py -3.11 -m venv .venv
.\.venv\Scripts\pip install -r requirements-dev.txt
copy .env.example .env
# Editar ROBOFLOW_API_KEY en .env
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

Luego levantar servicios:

```powershell
# Terminal 1 — API
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --app-dir . --port 8080

# Terminal 2 — Streamlit
$env:API_URL = "http://localhost:8080"
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py
```

---

## Fuentes de datos

- **DS1:** `fall-detection-raskl` v2 (Roboflow)
- **DS2:** `fa-nunl5` v1 (Roboflow)

Dataset fusionado: ~1.373 imágenes en `data/fused/`.

---

## Buenas prácticas ML aplicadas

- Balance de clases (`WeightedRandomSampler` + pesos en loss)
- Chequeo de data leakage entre splits
- Separación entrenamiento offline vs inferencia en API
- Checkpoint del mejor modelo por `val_acc`
- Fine-tuning en dos fases (capa final + `layer4`)
- Respuesta con **confianza** (`softmax`) en `/predict`

---

## Enlaces útiles

- [Swagger UI](http://localhost:8080/docs) (con API levantada)
- [Manual de ejecución local](../blob/marcelo/MANUAL_EJECUCION.txt) (en el repo)
- [Ejemplos API](../blob/marcelo/docs/api_examples.md)
