from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.analytics.db import init_db
from src.api.routers.dashboard import router as dashboard_router
from src.api.routers.predict import router as predict_router
from src.api.routers.gradcam import router as gradcam_router
from src.api.routers.health import health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Detector de Caídas API",
    description="Clasificación binaria Fall / Not Fall con ResNet18, explicabilidad Grad-CAM y analytics.",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "API activa. Usa /docs para ver los endpoints.",
        "endpoints": {
            "health": "/health",
            "predict_cnn": "POST /predict",
            "gradcam": "POST /gradcam",
            "dashboard_stats": "GET /dashboard/stats",
            "docs": "/docs",
        },
    }


app.include_router(health_router, tags=["Health"])
app.include_router(predict_router, tags=["CNN"])
app.include_router(gradcam_router, tags=["Explainability"])
app.include_router(dashboard_router)
