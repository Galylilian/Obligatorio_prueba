from fastapi import FastAPI

from .predict import router as predict_router
from .gradcam import router as gradcam_router
from .health import health_router
from .predict_video import router as predict_video_router
from .dashboard import router as dashboard_router


def init_routers(app: FastAPI) -> None:
    """
    Inicializa los routers de la API
    """

    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(gradcam_router)
    app.include_router(predict_video_router)
    app.include_router(dashboard_router)