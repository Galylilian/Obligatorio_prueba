# Instalación y configuración

[[← Inicio|Home]]

---

## Requisitos

| Requisito | Detalle |
|-----------|---------|
| Python | **3.11** (64 bits recomendado) |
| SO | Windows, Linux o macOS |
| Cuenta Roboflow | Para descargar datasets |
| Git | Opcional, para clonar el repo |

---

## Clonar el repositorio

```bash
git clone -b marcelo https://github.com/Galylilian/Obligatorio_prueba.git
cd Obligatorio_prueba
```

---

## Entorno virtual

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements-dev.txt
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

---

## Variables de entorno

```powershell
copy .env.example .env   # Windows
cp .env.example .env     # Linux/macOS
```

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `ROBOFLOW_API_KEY` | Clave API Roboflow | Obtener en [app.roboflow.com](https://app.roboflow.com) |
| `MODEL_PATH` | Pesos del modelo | `models/resnet18_best.pth` |
| `DATA_DIR` | Dataset fusionado | `data/fused` |
| `API_URL` | URL API (Streamlit) | `http://localhost:8080` |
| `APP_ENV` | Entorno | `development` |

Configuración central: `src/settings/config.py`.

---

## Regla importante

**Siempre** usar el Python del `.venv`:

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py   # ✅
python scripts\download_dataset.py                        # ❌ puede fallar
```

---

## Kernel Jupyter (para EDA)

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name=obligatorio --display-name="Python (obligatorio)"
```

En Cursor/VS Code: seleccionar kernel **Python (obligatorio)** al abrir `notebooks/EDA_Caidas.ipynb`.

---

## Verificar instalación

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import torch, roboflow, fastapi, streamlit; print('OK')"
```

---

## Qué NO se sube a GitHub

- `.env` (contiene API keys)
- `.venv/`
- `data/` (imágenes)
- `models/` (pesos entrenados)

Estos se generan localmente después de clonar.
