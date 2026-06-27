"""
reconstruction.py (schemas)
Modelos Pydantic para validar el request y documentar las respuestas.

Nota de compatibilidad: el frontend ACTUAL (reconstructionApi.js) envía
{ siteId, mlCategory, currentYear, targetYear }. Por eso solo `siteId` es
obligatorio; el resto de campos son opcionales (incluidos los del plan original
imageUrl/historicalInfo, por si se usan más adelante).
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ReconstructionRequest(BaseModel):
    siteId: str = Field(..., description="ID del sitio, p. ej. 'centro_lima'.")
    # Campos que envía el frontend actual:
    mlCategory: Optional[str] = None
    currentYear: Optional[int] = None
    targetYear: Optional[int] = None
    # Campos opcionales (plan original / uso futuro):
    siteName: Optional[str] = None
    imageUrl: Optional[str] = None
    historicalInfo: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "siteId": "centro_lima",
                "mlCategory": "urban_historical",
                "currentYear": 2077,
                "targetYear": 2010,
            }
        }
    }


class PredictionResult(BaseModel):
    # Se usa el alias "class" en el JSON (palabra reservada en Python).
    class_: str = Field(..., alias="class")
    confidence: float
    modelVersion: str

    model_config = {"populate_by_name": True}


class SiteSummary(BaseModel):
    siteId: str
    siteName: str
    category: str
    currentYear: int
    targetYear: int
    sceneTemplate: str


class SourceInfo(BaseModel):
    datasetVersion: str
    templateUsed: str


class ReconstructionResponse(BaseModel):
    """Forma de la respuesta de /reconstruct (documentación de Swagger)."""

    siteId: str
    sceneName: str
    currentYear: int
    targetYear: int
    prediction: PredictionResult
    environment: dict[str, Any]
    objects: list[dict[str, Any]]
    hotspots: list[dict[str, Any]]
    effects: dict[str, Any]
    source: SourceInfo
