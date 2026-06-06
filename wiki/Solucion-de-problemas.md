# Solución de problemas

[[← Inicio|Home]]

Cosas que nos pasaron (y cómo las arreglamos). Si tu error no está acá, revisá que estés usando el `.venv`.

---

## "No module named 'urllib3'" (o cualquier módulo)

**Causa:** Corriste `python` del sistema en vez del del proyecto.

**Arreglo:**
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts\download_dataset.py
```

---

## Roboflow no descarga

**"Falta ROBOFLOW_API_KEY"** → copiá `.env.example` a `.env` y pegá tu clave.

**Error SSL** → el script ya tiene un workaround; si sigue fallando, probá sin VPN o revisá el firewall.

---

## PowerShell no deja ejecutar `.ps1`

**Error:** "ejecución de scripts deshabilitada"

**Arreglo rápido** — corré los comandos a mano:
```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --app-dir . --port 8080
```

**Arreglo permanente** (solo tu usuario):
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## Puerto 8080 "no permitido" o ocupado

Casi siempre la API **ya está corriendo**. Abrí http://localhost:8080/docs — si carga, no necesitás levantar otra.

---

## Streamlit no muestra nada

1. ¿La API está en :8080? → `curl http://localhost:8080/health`
2. ¿`API_URL` apunta bien? → `$env:API_URL = "http://localhost:8080"`

---

## El modelo se equivoca con fotos de internet

**Es normal.** Entrenamos con Roboflow indoor, no con Getty Images. Para la demo usá `data/fused/test/`.

---

## Confianza baja (< 70%)

El modelo duda. Mirá el gráfico de probabilidades en Streamlit. No es bug — es señal de incertidumbre.

---

## Reentrené pero la API responde igual

Reiniciá uvicorn. La API cachea el modelo en memoria.

---

## Notebook: "ipykernel no instalado"

Elegí el kernel `.venv\Scripts\python.exe` o:
```powershell
.\.venv\Scripts\pip install ipykernel
.\.venv\Scripts\python.exe -m ipykernel install --user --name=obligatorio
```

---

## Git push falla por SSL

Workaround temporal:
```powershell
$env:GIT_SSL_NO_VERIFY='true'
git push obligatorio_prueba marcelo
```

---

## Checklist del "¿por qué no anda?"

- [ ] ¿Estoy en la carpeta del proyecto?
- [ ] ¿Uso `.\.venv\Scripts\python.exe`?
- [ ] ¿Existe `.env` con Roboflow key?
- [ ] ¿Hay `models/resnet18_best.pth`?
- [ ] ¿La API responde en `/health`?
- [ ] ¿Streamlit tiene `API_URL` correcta?

Si todo es sí y sigue fallando, revisá la terminal — el error suele estar ahí.
