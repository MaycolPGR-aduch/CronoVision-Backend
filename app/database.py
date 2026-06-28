"""
database.py
Configuración de la base de datos SQLite con SQLAlchemy (síncrono).

SQLite es ideal para este proyecto:
  - Sin servidor externo (el archivo .db se crea automáticamente).
  - Versionable o ignorable según necesidad.
  - Suficiente para el volumen de un MVP/demo.

El archivo se crea en la raíz del backend: chrono_vision.db
"""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Ruta del archivo SQLite relativa a este módulo
DB_PATH = Path(__file__).resolve().parent.parent / "chrono_vision.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# check_same_thread=False es necesario para que FastAPI (multithreaded) use SQLite.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # True para ver SQL en consola (útil al depurar)
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base declarativa para todos los modelos SQLAlchemy."""
    pass


def init_db() -> None:
    """Crea las tablas si no existen. Llamar al arrancar la app."""
    # Importa los modelos para que Base los registre antes de create_all.
    from app import models  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependencia FastAPI: abre y cierra la sesión por request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
