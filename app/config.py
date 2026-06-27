"""
config.py
Configuración central del backend Chrono-Vision: rutas, versión, CORS y el
mapeo de clase ML → plantilla de escena. Las rutas se resuelven respecto a
este archivo, así funcionan sin importar desde qué carpeta se ejecute.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # lee backend/.env si existe (variable FRONTEND_ORIGIN)

# ── Rutas ─────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent           # backend/app
DATA_DIR = APP_DIR / "data"
TEMPLATES_DIR = APP_DIR / "templates"
ARTIFACTS_DIR = APP_DIR / "ml" / "artifacts"

DATASET_PATH = DATA_DIR / "sites_dataset.json"
MODEL_PATH = ARTIFACTS_DIR / "reconstruction_classifier.joblib"

# ── Metadatos del servicio ────────────────────────────────────────────────────
SERVICE_NAME = "Chrono-Vision Reconstruction API"
SERVICE_VERSION = "1.0.0"
MODEL_VERSION = "tfidf-logreg-v1"

# ── CORS ──────────────────────────────────────────────────────────────────────
# Orígenes del frontend en desarrollo + uno extra configurable por entorno.
_DEFAULT_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")
CORS_ORIGINS = _DEFAULT_ORIGINS + ([FRONTEND_ORIGIN] if FRONTEND_ORIGIN else [])

# ── Mapeo clase ML → archivo de plantilla de escena ───────────────────────────
TEMPLATE_BY_CLASS = {
    "urban_historical": "urban_historical.json",
    "archaeological_zone": "archaeological_zone.json",
    "coastal_fortress": "coastal_fortress.json",
}
