# Streamlit

[[← Inicio|Home]] · [[API e inferencia]]

Streamlit es la **interfaz visual** para probar el detector sin escribir curl ni JSON. Ideal para la demo del obligatorio.

---

## Antes de empezar

La **API tiene que estar corriendo** en http://localhost:8080. Si no, Streamlit no tiene a quién preguntarle.

---

## Cómo levantarlo

```powershell
cd Obligatorio_prueba
$env:PYTHONPATH = (Get-Location).Path
$env:API_URL = "http://localhost:8080"
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py --server.port 8501
```

Abrí: **http://localhost:8501**

> **Tip Windows:** no ejecutes `streamlit_app.py` solo — siempre con `streamlit run app\streamlit_app.py`.

---

## Qué vas a ver

1. Botón para **subir una foto**
2. La imagen original
3. **Predicción** (CAÍDA / NO CAÍDA) con **porcentaje de confianza**
4. Gráfico de probabilidades
5. **Grad-CAM** debajo
6. Si la confianza es baja (< 70%), te avisa en amarillo

---

## ¿Quién hace el trabajo?

Streamlit **no** carga el modelo. Solo manda la foto a la API y muestra la respuesta:

```
Vos → Streamlit → API → ResNet18 → respuesta → pantalla
```

---

## Fotos para probar

Andá a estas carpetas y elegí cualquiera:

```
data/fused/test/fall/
data/fused/test/not_fall/
```

Evitá fotos de internet o de stock — el modelo se confunde. Con las del test set deberías ver confianzas altas (> 90%).

---

## Problemas típicos

| Te pasa esto | Probablemente… |
|--------------|----------------|
| No pasa nada al subir foto | La API no está levantada |
| Confianza muy baja | Foto fuera del tipo que entrenamos |
| Puerto 8501 ocupado | Usá `--server.port 8502` |

Más ayuda en [[Solución de problemas]].
