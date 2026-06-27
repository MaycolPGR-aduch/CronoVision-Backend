"""
reconstruction_service.py
Arma el JSON final de reconstrucción, compatible con el renderizador A-Frame
del frontend (objectFactory.js / reconstructionRenderer.js).

Combina:
  - metadatos del sitio (dataset)
  - la predicción del ML
  - environment / objects / hotspots / effects de la PLANTILLA elegida por la
    clase predicha (las plantillas son copia de la escena ya verificada).
"""

import json
from functools import lru_cache

from app.config import TEMPLATES_DIR, TEMPLATE_BY_CLASS
from app.services import dataset_service


class TemplateError(Exception):
    """Error al resolver/cargar la plantilla de escena."""


@lru_cache(maxsize=8)
def _load_template(filename: str) -> dict:
    path = TEMPLATES_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_reconstruction(site: dict, prediction: dict) -> dict:
    pred_class = prediction.get("class")
    # La plantilla la elige la clase PREDICHA por el ML (así el modelo "decide"
    # la reconstrucción). Si esa clase no tuviera plantilla, cae a la del sitio.
    template_file = TEMPLATE_BY_CLASS.get(pred_class) or TEMPLATE_BY_CLASS.get(
        site.get("category")
    )
    if not template_file:
        raise TemplateError(f"No hay plantilla para la clase '{pred_class}'.")

    try:
        tpl = _load_template(template_file)
    except FileNotFoundError as exc:
        raise TemplateError(f"No se encontró la plantilla '{template_file}'.") from exc
    except json.JSONDecodeError as exc:
        raise TemplateError(f"Plantilla '{template_file}' inválida.") from exc

    return {
        "siteId": site["siteId"],
        "sceneName": tpl.get("sceneName") or f"{site['siteName']} reconstruido",
        "currentYear": site["currentYear"],
        "targetYear": site["targetYear"],
        "prediction": {
            "class": prediction["class"],
            "confidence": prediction["confidence"],
            "modelVersion": prediction["modelVersion"],
        },
        "environment": tpl.get("environment", {}),
        "objects": tpl.get("objects", []),
        "hotspots": tpl.get("hotspots", []),
        "effects": tpl.get("effects", {}),
        "source": {
            "datasetVersion": dataset_service.get_dataset_version(),
            "templateUsed": template_file,
        },
    }
