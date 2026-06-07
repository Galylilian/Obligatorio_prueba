# Grafana y plataforma analítica

[[← Inicio|Home]] · [[Docker]]

Además de predecir caídas, el proyecto puede **registrar eventos** y mostrar KPIs en **Grafana**. No son pacientes clínicos: son **personas** monitoreadas (`person_id`).

---

## ¿Qué problema resuelve?

| Solo inferencia | Con analytics |
|-----------------|---------------|
| `POST /predict` → respuesta y listo | Cada predicción se **guarda** |
| No hay historial | Podés ver tendencias en el tiempo |
| No hay dashboard | Grafana con KPIs en vivo |

---

## Componentes

```
POST /predict (+ person_id)
        ↓
   PostgreSQL  ←  prediction_events
        ↓
   Grafana :3000  (gráficos)
        ↓
GET /dashboard/stats  (JSON para otras herramientas)
```

---

## KPIs disponibles

**`GET /dashboard/stats`** devuelve:

```json
{
  "falls_today": 12,
  "falls_week": 43,
  "high_risk_persons": 5,
  "analytics_enabled": true
}
```

| Campo | Significado |
|-------|-------------|
| `falls_today` | Caídas detectadas hoy (UTC) |
| `falls_week` | Caídas en los últimos 7 días |
| `high_risk_persons` | Personas con ≥ 2 caídas en la semana |

La regla de alto riesgo se configura con `HIGH_RISK_MIN_FALLS_WEEK` en `.env` (default: 2).

---

## Levantar todo con Docker

**Requisito:** Docker Desktop instalado y corriendo.

```powershell
cd Obligatorio_prueba
powershell -ExecutionPolicy Bypass -File .\scripts\start_analytics.ps1
```

O manualmente:

```powershell
$env:DATABASE_URL = "postgresql+psycopg2://fallapp:fallapp@db:5432/fall_analytics"
docker compose --profile analytics up --build -d db api grafana
```

| Servicio | URL | Credenciales |
|----------|-----|--------------|
| Grafana | http://localhost:3000 | admin / admin |
| API stats | http://localhost:8080/dashboard/stats | — |
| Swagger | http://localhost:8080/docs | — |
| PostgreSQL | localhost:5432 | fallapp / fallapp |

Dashboard precargado: **Deteccion de caidas - KPIs**.

---

## Registrar eventos (personas)

Al llamar `/predict`, mandá el ID de la persona:

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@imagen.jpg" \
  -F "person_id=P042" \
  -F "source=camera_lobby"
```

- **`person_id`** — identifica a la persona (no es paciente clínico)
- **`source`** — cámara, upload, etc. (opcional)

Cada predicción queda en la tabla `prediction_events` con timestamp, label y confianza.

---

## Grafana: qué verás

El dashboard incluye:

1. **Caídas hoy** — número grande
2. **Caídas última semana**
3. **Personas en alto riesgo**
4. **Gráfico de caídas por hora** (últimos 7 días)

Grafana lee **directo de PostgreSQL** (datasource `FallAnalytics` provisionado automáticamente).

---

## Sin Docker (solo API local)

Si corrés la API sin Postgres, todo sigue funcionando excepto analytics:

- `/predict` responde normal
- `/dashboard/stats` devuelve ceros y `"analytics_enabled": false`

Para analytics local sin Docker necesitás PostgreSQL instalado y en `.env`:

```
DATABASE_URL=postgresql+psycopg2://fallapp:fallapp@localhost:5432/fall_analytics
```

---

## Instalar Docker Desktop (Windows)

Si `docker` no se reconoce en PowerShell:

1. Descargá [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Instalá y reiniciá la PC si lo pide
3. Abrí Docker Desktop (ícono de ballena en la bandeja)
4. Volvé a correr `start_analytics.ps1`

---

## Para el informe (Caso 7)

Podés escribir:

> *“Implementamos una capa analítica sobre la inferencia: cada predicción alimenta PostgreSQL. Grafana expone KPIs operacionales (caídas por día/semana, personas en alto riesgo). El endpoint `/dashboard/stats` permite integrar los mismos datos en Power BI o Superset.”*

---

## Archivos relevantes en el repo

| Ruta | Rol |
|------|-----|
| `src/analytics/` | Modelos SQLAlchemy + consultas |
| `src/api/routers/dashboard.py` | `/dashboard/stats` |
| `grafana/dashboards/` | Dashboard JSON |
| `grafana/provisioning/` | Datasource Postgres auto |
| `docker-compose.yml` | Perfil `analytics` |
| `scripts/start_analytics.ps1` | Arranque rápido |
