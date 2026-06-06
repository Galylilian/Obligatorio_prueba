# Solución de problemas

[[← Inicio|Home]]

---

## Python / dependencias

### `No module named 'urllib3'` (o cualquier módulo)

**Causa:** Se usó `python` del sistema en lugar del `.venv`.

**Solución:**
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts\download_dataset.py
```

---

## Roboflow

### `Falta ROBOFLOW_API_KEY en .env`

Copiar `.env.example` → `.env` y pegar la clave de [Roboflow](https://app.roboflow.com).

### Error SSL al descargar

El script `download_dataset.py` incluye workaround SSL. Si persiste: revisar firewall/VPN/proxy.

---

## PowerShell

### `Ejecución de scripts deshabilitada` (.ps1)

Usar comandos directos en lugar de scripts:

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --app-dir . --port 8080
```

O habilitar scripts (solo tu usuario):
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## Puertos

### `[WinError 10013]` puerto 8080

**Causa:** La API ya está corriendo.

**Solución:** Usar la instancia existente en http://localhost:8080/docs o cerrar el proceso anterior.

### Streamlit puerto 8501 ocupado

```powershell
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py --server.port 8502
$env:API_URL = "http://localhost:8080"
```

---

## Modelo / predicciones

### API devuelve predicciones viejas tras reentrenar

Reiniciar uvicorn (Ctrl+C y volver a levantar).

### Falso positivo con foto de internet

Normal: el modelo fue entrenado con Roboflow indoor. Probar con `data/fused/test/`.

### Confianza baja (< 70%)

El modelo no está seguro. Revisar calidad de imagen y dominio. Ver barra de probabilidades en Streamlit.

### Grad-CAM confuso

El heatmap es aproximado y la imagen se reduce a 224×224. Ver [[Modelo y entrenamiento#Limitaciones conocidas]].

---

## Jupyter / notebook

### `ipykernel no instalado`

Seleccionar kernel `.venv\Scripts\python.exe` o instalar:
```powershell
.\.venv\Scripts\pip install ipykernel
.\.venv\Scripts\python.exe -m ipykernel install --user --name=obligatorio
```

---

## Git / GitHub

### Error SSL con `git push`

```powershell
$env:GIT_SSL_NO_VERIFY='true'
git push obligatorio_prueba marcelo
```

(Solo workaround temporal; ideal arreglar certificados del sistema.)

---

## Checklist rápido

- [ ] Python 3.11 + `.venv` activo
- [ ] `requirements-dev.txt` instalado
- [ ] `.env` con `ROBOFLOW_API_KEY`
- [ ] `models/resnet18_best.pth` existe
- [ ] API responde en `/health`
- [ ] Streamlit apunta a `API_URL` correcta
