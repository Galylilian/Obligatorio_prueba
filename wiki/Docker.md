# Docker

[[← Inicio|Home]]

---

## Requisitos

- Docker Desktop instalado y corriendo
- Modelo entrenado en `models/resnet18_best.pth` (montado como volumen)

---

## Solo API

```bash
docker compose up --build
```

API: **http://localhost:8080**

El contenedor monta `./models` como solo lectura.

---

## API + Streamlit

```bash
docker compose --profile full up --build
```

| Servicio | Puerto |
|----------|--------|
| API | 8080 |
| Streamlit | 8501 |

---

## Detener

```bash
docker compose down
```

Con volúmenes:

```bash
docker compose down -v
```

---

## Variables en Docker

Definidas en `docker-compose.yml`:

- `APP_ENV=production`
- `MODEL_PATH=/app/models/resnet18_best.pth`
- `API_URL=http://api:8080` (Streamlit → API interna)

---

## Notas

- El Dockerfile de producción usa `requirements.txt` (sin Roboflow/jupyter).
- Entrenamiento siempre es **offline** en la máquina host, no dentro del contenedor de inferencia.
- Copiar `models/` al servidor antes de `docker compose up`.
