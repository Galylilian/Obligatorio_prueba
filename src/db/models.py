from sqlalchemy import Column, Integer, String, Float, DateTime, func
from src.db.database import Base


class Prediction(Base):
    """
    Tabla que registra cada predicción realizada por la API.
    Permite contar imágenes clasificadas, ver distribución de labels,
    y alimentar el dashboard de Grafana.
    """

    __tablename__ = "predictions"

    id          = Column(Integer, primary_key=True, index=True)
    filename    = Column(String, nullable=True)           # nombre del archivo subido
    label       = Column(String, nullable=False)           # "fall" o "no_fall"
    confidence  = Column(Float, nullable=False)            # confianza del modelo (0-1)
    model_type  = Column(String, nullable=False)           # "normal" o "quantized"
    created_at  = Column(DateTime(timezone=True), server_default=func.now())