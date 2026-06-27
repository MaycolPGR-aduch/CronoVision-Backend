"""
routes.py
Endpoints de la API Chrono-Vision:
  GET  /                 → estado del servicio
  GET  /sites            → lista resumida de sitios
  GET  /sites/{site_id}  → info completa del sitio (404 si no existe)
  POST /reconstruct      → predicción ML + JSON de reconstrucción A-Frame
"""

from fastapi import APIRouter, HTTPException

from app.config import SERVICE_NAME, SERVICE_VERSION
from app.schemas.reconstruction import ReconstructionRequest, SiteSummary
from app.services import dataset_service, ml_service, reconstruction_service

router = APIRouter()


@router.get("/", tags=["meta"])
def root():
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@router.get("/sites", response_model=list[SiteSummary], tags=["sites"])
def list_sites():
    return dataset_service.get_site_summaries()


@router.get("/sites/{site_id}", tags=["sites"])
def get_site(site_id: str):
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
    site = dataset_service.get_site(req.siteId)
    if site is None:
        raise HTTPException(status_code=404, detail=f"Sitio '{req.siteId}' no encontrado.")

    prediction = ml_service.predict_reconstruction(site, req)

    try:
        return reconstruction_service.build_reconstruction(site, prediction)
    except reconstruction_service.TemplateError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
