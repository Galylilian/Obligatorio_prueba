from fastapi import FastAPI

from src.api.routers.predict import router as predict_router
from src.api.routers.gradcam import router as gradcam_router
from src.api.routers.health import health_router

app = FastAPI(
    title="Detector de Caídas API",
    description="Clasificación binaria Fall / Not Fall con ResNet18 y explicabilidad Grad-CAM.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "API activa. Usa /docs para ver los endpoints.",
        "endpoints": {
            "health": "/health",
            "predict_cnn": "POST /predict",
            "gradcam": "POST /gradcam",
            "docs": "/docs",
        },
    }


app.include_router(health_router, tags=["Health"])
app.include_router(predict_router, tags=["CNN"])
app.include_router(gradcam_router, tags=["Explainability"])
