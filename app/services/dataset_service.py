"""
dataset_service.py
Acceso al dataset (backend/app/data/sites_dataset.json): sitios, resúmenes,
muestras de entrenamiento y versión. Se cachea en memoria con lru_cache.
"""

import json
from functools import lru_cache

from app.config import DATASET_PATH


@lru_cache(maxsize=1)
def load_dataset() -> dict:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_dataset_version() -> str:
    return load_dataset().get("datasetVersion", "1.0.0")


def get_sites() -> list[dict]:
    return load_dataset().get("sites", [])


def get_site(site_id: str) -> dict | None:
    for site in get_sites():
        if site.get("siteId") == site_id:
            return site
    return None


def get_site_summaries() -> list[dict]:
    """Lista resumida para GET /sites."""
    return [
        {
            "siteId": s["siteId"],
            "siteName": s["siteName"],
            "category": s["category"],
            "currentYear": s["currentYear"],
            "targetYear": s["targetYear"],
            "sceneTemplate": s.get("sceneTemplate", ""),
        }
        for s in get_sites()
    ]


def get_training_samples() -> list[dict]:
    return load_dataset().get("trainingSamples", [])
