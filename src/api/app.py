from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
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


def custom_openapi():
    """Agrega 'format: binary' a los campos de archivo.

    FastAPI genera OpenAPI 3.1, que describe los uploads con
    'contentMediaType' en vez de 'format: binary'. Swagger UI no
    reconoce 'contentMediaType' en items de un array, por lo que
    renderiza `list[UploadFile]` como un array de strings en vez
    de un selector de archivos.
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    for component in schema.get("components", {}).get("schemas", {}).values():
        for prop in component.get("properties", {}).values():
            if prop.get("contentMediaType") == "application/octet-stream":
                prop["format"] = "binary"
            items = prop.get("items")
            if isinstance(items, dict) and items.get("contentMediaType") == "application/octet-stream":
                items["format"] = "binary"

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi