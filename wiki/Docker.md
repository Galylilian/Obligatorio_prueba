# Docker

[[← Inicio|Home]]

Docker empaqueta la API (y opcionalmente Streamlit) para correr igual en cualquier servidor — sin pelear con Python en cada máquina.

---

## Antes de arrancar

- Docker Desktop instalado y corriendo
- Modelo ya entrenado en `models/resnet18_best.pth` (Docker **no** entrena, solo sirve predicciones)

---

## Solo la API

```bash
docker compose up --build
```

→ http://localhost:8080

El contenedor lee `./models` de tu disco (solo lectura).

---

## API + Streamlit juntos

```bash
docker compose --profile full up --build
```

| Servicio | Puerto |
|----------|--------|
| API | 8080 |
| Streamlit | 8501 |

---

## Para apagar

```bash
docker compose down
```

---

## Idea clave

| Qué | Dónde corre |
|-----|-------------|
| Entrenar (`train.py`) | Tu PC, offline |
| Predecir (API) | Docker en producción |

Entrenás una vez, copiás `models/` al servidor, levantás Docker. Eso es MLP en producción 🚀
