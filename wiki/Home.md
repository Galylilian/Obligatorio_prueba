# Bienvenido 👋

Este es el proyecto del obligatorio de **Machine Learning en Producción** (ORT Uruguay): un detector de caídas que mira una foto y responde si la persona **cayó** o **no**.

**Repo:** [Obligatorio_prueba · rama `marcelo`](https://github.com/Galylilian/Obligatorio_prueba/tree/marcelo)

---

## ¿Qué hace en pocas palabras?

Le das una imagen → el sistema te dice:

| Resultado | Significa |
|-----------|-----------|
| `fall` | Parece una caída |
| `not_fall` | Persona de pie, sentada o en situación normal |

Además tenés:
- Un **pipeline** para bajar datos, entrenar y evaluar el modelo
- Una **API** (FastAPI) para usar el modelo desde cualquier app
- **Streamlit** para probarlo subiendo fotos desde el navegador
- **Grad-CAM** para ver *dónde* miró la red al decidir
- Un **notebook EDA** para el informe

---

## ¿Cómo está armado?

Piensalo así:

```
Roboflow (internet)  →  entrenamos offline  →  guardamos el modelo
                                                      ↓
                              Vos subís una foto  →  API  →  respuesta
                                                      ↑
                                              Streamlit (interfaz bonita)
```

---

## Guías de esta wiki

| Si querés… | Andá a… |
|------------|---------|
| Instalar todo desde cero | [[Instalación y configuración]] |
| Correr download, fuse, train… | [[Pipeline offline]] |
| Entender el modelo ResNet18 | [[Modelo y entrenamiento]] |
| Usar `/predict` y `/gradcam` | [[API e inferencia]] |
| Abrir la interfaz web | [[Streamlit]] |
| Hacer el EDA del informe | [[EDA y datos]] |
| Desplegar con Docker | [[Docker]] |
| Algo falló 😅 | [[Solución de problemas]] |

---

## Arranque rápido (5 comandos)

```powershell
git clone -b marcelo https://github.com/Galylilian/Obligatorio_prueba.git
cd Obligatorio_prueba
py -3.11 -m venv .venv
.\.venv\Scripts\pip install -r requirements-dev.txt
copy .env.example .env
```

Editá `.env` con tu `ROBOFLOW_API_KEY`, y después:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

Cuando termine el entrenamiento, en **dos terminales**:

```powershell
# Terminal 1 — API
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --app-dir . --port 8080

# Terminal 2 — Streamlit
$env:API_URL = "http://localhost:8080"
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py
```

---

## De dónde salen las fotos

Usamos dos datasets de Roboflow que fusionamos:
- **DS1:** `fall-detection-raskl` (v2)
- **DS2:** `fa-nunl5` (v1)

En total quedan unas **1.373 imágenes** en `data/fused/`.

---

## Cosas que hicimos bien (para el informe)

- Balanceamos clases para que el modelo no se sesgue
- Revisamos que no haya **data leakage** entre train y test
- Separamos entrenamiento (offline) de inferencia (API)
- Guardamos el mejor modelo según validación
- Entrenamos en **dos fases** (capa final + fine-tuning)
- La API devuelve **confianza** además de la etiqueta

---

## Links útiles

- [Swagger](http://localhost:8080/docs) — probá la API en el navegador
- [Manual de ejecución](../blob/marcelo/MANUAL_EJECUCION.txt) — paso a paso en txt
- [Ejemplos curl](../blob/marcelo/docs/api_examples.md)
