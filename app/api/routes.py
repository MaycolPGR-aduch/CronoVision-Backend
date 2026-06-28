"""
routes.py
Endpoints de la API Chrono-Vision:
  GET  /                      → estado del servicio + salud del modelo ML
  GET  /sites                 → lista resumida de sitios
  GET  /sites/{site_id}       → info completa del sitio (404 si no existe)
  POST /reconstruct           → predicción ML + JSON de reconstrucción A-Frame
  GET  /narrate/{site_id}     → narración histórica con streaming (Groq LLaMA 3)
"""

import re
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.config import SERVICE_NAME, SERVICE_VERSION
from app.schemas.reconstruction import ReconstructionRequest, SiteSummary
from app.services import dataset_service, ml_service, reconstruction_service, narration_service

router = APIRouter()

# Patrón válido para siteId: solo letras minúsculas, números y guion bajo.
_SITE_ID_RE = re.compile(r'^[a-z0-9_]{1,64}$')


def _validate_site_id(site_id: str) -> None:
    """Lanza 400 si el siteId tiene caracteres inválidos."""
    if not _SITE_ID_RE.match(site_id):
        raise HTTPException(status_code=400, detail="siteId inválido.")


@router.get("/", tags=["meta"])
def root():
    """Health check: incluye estado del modelo ML para la demo."""
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
def reconstruct(req: ReconstructionRequest):
    """
    Recibe el sitio, corre el clasificador ML y devuelve el JSON de
    reconstrucción compatible con el frontend A-Frame.
    """
    _validate_site_id(req.siteId)
    site = dataset_service.get_site(req.siteId)
    if site is None:
        raise HTTPException(status_code=404, detail=f"Sitio '{req.siteId}' no encontrado.")

    prediction = ml_service.predict_reconstruction(site, req)

    try:
        return reconstruction_service.build_reconstruction(site, prediction)
    except reconstruction_service.TemplateError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/narrate/{site_id}", tags=["narration"])
def narrate(
    site_id: str,
    year: int | None = Query(default=None, ge=1, le=9999, description="Año de la reconstrucción"),
):
    """
    Genera una narración histórica inmersiva del sitio usando Groq (LLaMA 3)
    con streaming de texto. El frontend la recibe chunk a chunk y la muestra
    letra por letra en el panel.
    """
    _validate_site_id(site_id)
    site = dataset_service.get_site(site_id)
    if site is None:
        raise HTTPException(status_code=404, detail=f"Sitio '{site_id}' no encontrado.")

    target_year = year or site.get("targetYear", 2000)

    return StreamingResponse(
        narration_service.stream_narration(site, target_year),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )
