"""
history_service.py
Operaciones de base de datos para el historial de reconstrucciones.

Estrategia robusta:
  - Nunca lanza excepciones hacia el caller (la API no debe fallar por un
    error de BD).
  - Upsert silencioso del sitio antes de insertar la reconstrucción
    (garantiza integridad referencial sin requerir una carga previa).
"""

from __future__ import annotations

from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.models import Reconstruction, Site


def upsert_site(db: Session, site: dict) -> None:
    """
    Inserta el sitio si no existe; si ya existe, no hace nada (INSERT OR IGNORE).
    Así no duplicamos sitios pero tampoco fallamos si ya están.
    """
    stmt = (
        sqlite_insert(Site)
        .values(
            site_id      = site["siteId"],
            site_name    = site.get("siteName", ""),
            category     = site.get("category", ""),
            current_year = site.get("currentYear"),
            target_year  = site.get("targetYear"),
        )
        .on_conflict_do_nothing(index_elements=["site_id"])
    )
    db.execute(stmt)
    db.commit()


def save_reconstruction(db: Session, site: dict, prediction: dict) -> Reconstruction | None:
    """
    Guarda una reconstrucción en el historial.
    Devuelve el objeto creado o None si falla silenciosamente.
    """
    try:
        upsert_site(db, site)

        record = Reconstruction(
            site_id       = site["siteId"],
            pred_class    = prediction["class"],
            confidence    = prediction["confidence"],
            model_version = prediction.get("modelVersion"),
            target_year   = site.get("targetYear"),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    except Exception as exc:
        db.rollback()
        print(f"[history_service] Error al guardar reconstrucción: {exc}")
        return None


def get_recent_reconstructions(db: Session, limit: int = 20) -> list[dict]:
    """
    Devuelve las últimas `limit` reconstrucciones con info del sitio.
    Ordenadas por fecha descendente.
    """
    rows = (
        db.query(Reconstruction)
        .order_by(Reconstruction.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":           r.id,
            "siteId":       r.site_id,
            "siteName":     r.site.site_name if r.site else r.site_id,
            "predClass":    r.pred_class,
            "confidence":   round(r.confidence, 4),
            "modelVersion": r.model_version,
            "targetYear":   r.target_year,
            "createdAt":    r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


def get_site_stats(db: Session) -> list[dict]:
    """
    Estadísticas por sitio: total de reconstrucciones y confianza promedio.
    """
    from sqlalchemy import func

    rows = (
        db.query(
            Reconstruction.site_id,
            func.count(Reconstruction.id).label("total"),
            func.avg(Reconstruction.confidence).label("avg_confidence"),
        )
        .group_by(Reconstruction.site_id)
        .all()
    )
    return [
        {
            "siteId":         r.site_id,
            "total":          r.total,
            "avgConfidence":  round(r.avg_confidence or 0, 4),
        }
        for r in rows
    ]
