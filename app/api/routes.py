"""
routes.py
Endpoints de la API Chrono-Vision:
  GET  /                      → estado del servicio + salud del modelo ML
  GET  /sites                 → lista resumida de sitios
  GET  /sites/{site_id}       → info completa del sitio (404 si no existe)
  POST /reconstruct           → predicción ML + JSON de reconstrucción A-Frame
  GET  /narrate/{site_id}     → narración histórica con streaming (Groq LLaMA 3)
  GET  /history               → últimas reconstrucciones (BD SQLite)
  GET  /history/stats         → estadísticas por sitio (BD SQLite)
"""

import re
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import SERVICE_NAME, SERVICE_VERSION
from app.database import get_db
from app.schemas.reconstruction import ReconstructionRequest, SiteSummary
from app.services import (
    dataset_service,
    ml_service,
    reconstruction_service,
    narration_service,
    history_service,
)

router = APIRouter()

_SITE_ID_RE = re.compile(r'^[a-z0-9_]{1,64}$')


def _validate_site_id(site_id: str) -> None:
    if not _SITE_ID_RE.match(site_id):
        raise HTTPException(status_code=400, detail="siteId inválido.")


@router.get("/", tags=["meta"])
def root():
    """Health check: incluye estado del modelo ML."""
    model_ok = ml_service._get_model() is not None
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "model": {
            "loaded": model_ok,
            "version": SERVICE_VERSION if model_ok else "fallback-rules-v1",
        },
    }


@router.get("/sites", response_model=list[SiteSummary], tags=["sites"])
def list_sites():
    return dataset_service.get_site_summaries()


@router.get("/sites/{site_id}", tags=["sites"])
def get_site(site_id: str):
    _validate_site_id(site_id)
    site = dataset_service.get_site(site_id)
    if site is None:
        raise HTTPException(status_code=404, detail=f"Sitio '{site_id}' no encontrado.")
    return site


@router.post("/reconstruct", tags=["reconstruction"])
def reconstruct(req: ReconstructionRequest, db: Session = Depends(get_db)):
    """
    Clasifica el sitio con ML, devuelve el JSON de escena A-Frame
    y guarda el resultado en el historial SQLite.
    """
    _validate_site_id(req.siteId)
    site = dataset_service.get_site(req.siteId)
    if site is None:
        raise HTTPException(status_code=404, detail=f"Sitio '{req.siteId}' no encontrado.")

    prediction = ml_service.predict_reconstruction(site, req)

    try:
        result = reconstruction_service.build_reconstruction(site, prediction)
    except reconstruction_service.TemplateError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Guarda en historial de forma silenciosa (no rompe la respuesta si falla).
    history_service.save_reconstruction(db, site, prediction)

    return result


@router.get("/narrate/{site_id}", tags=["narration"])
def narrate(
    site_id: str,
    year: int | None = Query(default=None, ge=1, le=9999, description="Año de la reconstrucción"),
):
    """Narración histórica con streaming (Groq LLaMA 3)."""
    _validate_site_id(site_id)
    site = dataset_service.get_site(site_id)
    if site is None:
        raise HTTPException(status_code=404, detail=f"Sitio '{site_id}' no encontrado.")

    target_year = year or site.get("targetYear", 2000)

    return StreamingResponse(
        narration_service.stream_narration(site, target_year),
        media_type="text/plain; charset=utf-8",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@router.get("/history", tags=["history"])
def get_history(
    limit: int = Query(default=20, ge=1, le=100, description="Número de registros"),
    db: Session = Depends(get_db),
):
    """
    Últimas reconstrucciones registradas en la BD SQLite.
    Útil para auditoría, demo y estadísticas.
    """
    return history_service.get_recent_reconstructions(db, limit=limit)


@router.get("/history/stats", tags=["history"])
def get_history_stats(db: Session = Depends(get_db)):
    """
    Estadísticas por sitio: total de reconstrucciones y confianza promedio del modelo.
    """
    return history_service.get_site_stats(db)
