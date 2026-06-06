# Streamlit

[[← Inicio|Home]] · [[API e inferencia]]

---

## Requisito

La API debe estar corriendo en **http://localhost:8080** antes de abrir Streamlit.

---

## Levantar (Windows)

```powershell
cd Obligatorio_prueba
$env:PYTHONPATH = (Get-Location).Path
$env:API_URL = "http://localhost:8080"
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py --server.port 8501
```

Abrir: **http://localhost:8501**

---

## Qué muestra

1. Subir imagen (JPG/PNG)
2. Imagen original
3. **Predicción** (`fall` / `not_fall`) con **confianza %**
4. Gráfico de probabilidades por clase
5. **Grad-CAM** (mapa de calor)
6. JSON completo de la respuesta API

Si confianza < 70%, aparece advertencia.

---

## Arquitectura

```
Navegador → Streamlit :8501 → HTTP → API :8080 → ResNet18
```

Streamlit **no** carga el modelo directamente; delega en la API.

---

## Imágenes de prueba recomendadas

```
data/fused/test/fall/
data/fused/test/not_fall/
```

Evitar fotos de stock o de internet para la demo — el modelo fue entrenado con Roboflow indoor.

---

## Error común

| Síntoma | Causa | Solución |
|---------|-------|----------|
| No hay predicción | API caída | Levantar uvicorn primero |
| Confianza baja | Imagen fuera de dominio | Usar imágenes del test set |
| Puerto ocupado | Otra instancia Streamlit | Usar `--server.port 8502` |
