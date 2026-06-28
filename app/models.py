"""
models.py
Modelos SQLAlchemy — tablas de la base de datos relacional.

Diagrama ERD simplificado:

  sites ──────────────────────────────────────────
  │ site_id (PK)  │ site_name │ category │ ...   │
  ─────────────────────────────────────────────────
       │ 1
       │
       │ N
  reconstructions ───────────────────────────────────────────────────
  │ id (PK) │ site_id (FK) │ pred_class │ confidence │ created_at │
  ────────────────────────────────────────────────────────────────────

Relación: un sitio puede tener muchas reconstrucciones (1:N).
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.database import Base


class Site(Base):
    """
    Tabla de sitios históricos.
    Se puebla automáticamente desde el dataset JSON al hacer la primera
    reconstrucción de cada sitio (upsert silencioso).
    """
    __tablename__ = "sites"

    site_id      = Column(String(64),  primary_key=True, index=True)
    site_name    = Column(String(256), nullable=False)
    category     = Column(String(64),  nullable=False)
    current_year = Column(Integer,     nullable=True)
    target_year  = Column(Integer,     nullable=True)

    # Relación inversa: acceder a las reconstrucciones de un sitio.
    reconstructions = relationship(
        "Reconstruction",
        back_populates="site",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Site {self.site_id!r} ({self.category})>"


class Reconstruction(Base):
    """
    Tabla de historial de reconstrucciones.
    Cada llamada exitosa a POST /reconstruct inserta un registro.
    """
    __tablename__ = "reconstructions"

    id           = Column(Integer,     primary_key=True, autoincrement=True)
    site_id      = Column(String(64),  ForeignKey("sites.site_id"), nullable=False, index=True)
    pred_class   = Column(String(64),  nullable=False)
    confidence   = Column(Float,       nullable=False)
    model_version = Column(String(64), nullable=True)
    target_year  = Column(Integer,     nullable=True)
    created_at   = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relación hacia el sitio padre.
    site = relationship("Site", back_populates="reconstructions")

    def __repr__(self) -> str:
        return (
            f"<Reconstruction id={self.id} site={self.site_id!r} "
            f"class={self.pred_class!r} conf={self.confidence:.2f}>"
        )
