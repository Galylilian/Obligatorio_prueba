from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.api.routers.predict import router as predict_router
from src.api.routers.gradcam import router as gradcam_router
from src.api.routers.health import health_router
from src.api.routers.predict_video import router as predict_video_router
from src.api.routers.dashboard import router as dashboard_router
from src.db.database import init_db


# =============================
# INICIALIZAR DB AL ARRANCAR ✅
# =============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Fall Detector API",
    description="API para clasificación de caídas usando ResNet18",
    version="1.0.0",
    lifespan=lifespan,
)

# incluir routers
app.include_router(health_router, tags=["Health"])
app.include_router(predict_video_router, tags=["Video Prediction"])
app.include_router(predict_router, tags=["CNN"])
app.include_router(gradcam_router, tags=["Explainability"])
app.include_router(dashboard_router, tags=["Dashboard"])