# Instalación y configuración

[[← Volver al inicio|Home]]

Acá te explicamos cómo dejar el proyecto andando en tu máquina. No hace falta ser experto en DevOps: seguí los pasos en orden.

---

## ¿Qué necesitás?

- **Python 3.11** (64 bits, si podés)
- Una cuenta en **Roboflow** (gratis) para bajar los datasets
- **Git** si vas a clonar el repo
- Windows, Linux o macOS — probamos sobre todo en Windows

---

## 1. Clonar el repo

```bash
git clone -b marcelo https://github.com/Galylilian/Obligatorio_prueba.git
cd Obligatorio_prueba
```

---

## 2. Crear el entorno virtual

**Windows:**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\pip install -r requirements-dev.txt
```

**Linux / macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Tarda unos minutos la primera vez (descarga PyTorch, FastAPI, etc.).

---

## 3. Configurar el `.env`

```powershell
copy .env.example .env
```

Abrí `.env` y pegá tu clave de Roboflow:

```
ROBOFLOW_API_KEY=tu_clave_aqui
```

La sacás de [app.roboflow.com](https://app.roboflow.com) → Settings → API Key.

| Variable | Para qué sirve |
|----------|----------------|
| `ROBOFLOW_API_KEY` | Descargar datasets |
| `MODEL_PATH` | Dónde está el `.pth` entrenado |
| `API_URL` | Streamlit sabe a qué API pegarle |
| `DATA_DIR` | Carpeta `data/fused` |

Todo centralizado también en `src/settings/config.py`.

---

## ⚠️ Regla de oro

**Usá siempre el Python del `.venv`**, no el del sistema:

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py   ✅
python scripts\download_dataset.py                        ❌ suele fallar
```

Si ves `No module named 'urllib3'`, casi seguro es porque corriste el segundo.

---

## 4. Notebook EDA (opcional)

Si vas a usar `notebooks/EDA_Caidas.ipynb` en Cursor o VS Code:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name=obligatorio --display-name="Python (obligatorio)"
```

Elegí ese kernel al abrir el notebook.

---

## 5. ¿Quedó bien?

```powershell
.\.venv\Scripts\python.exe --version          # Python 3.11.x
.\.venv\Scripts\python.exe -c "import torch; print('OK')"
```

Si imprime `OK`, seguí con [[Pipeline offline]].

---

## Qué NO va a GitHub (y está bien)

- `.env` — tiene tu API key
- `.venv/` — tu entorno local
- `data/` — las imágenes pesan mucho
- `models/` — los pesos se generan al entrenar

Cada persona los crea en su PC después de clonar.
