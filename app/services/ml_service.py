"""
ml_service.py
Capa de Machine Learning REAL (clasificación de la escena).

Carga el modelo TF-IDF + LogisticRegression entrenado (artifacts/.joblib) y
predice la clase + confianza a partir del texto del sitio. Si el modelo no
existe (todavía no se entrenó) o falla, usa un FALLBACK basado en
`site["category"]` para no romper la API.

El frontend actual solo envía `siteId`, así que el texto para inferir se arma
desde el registro del sitio en el dataset (historicalInfo + visualTags + nombre).
"""

import joblib

from app.config import MODEL_PATH, MODEL_VERSION

_model = None
_load_attempted = False


def _get_model():
    """Carga perezosa del modelo; solo intenta una vez."""
    global _model, _load_attempted
    if not _load_attempted:
        _load_attempted = True
        try:
            if MODEL_PATH.exists():
                _model = joblib.load(MODEL_PATH)
        except Exception as exc:  # nunca debe romper la API
            print(f"[ml_service] No se pudo cargar el modelo: {exc}")
            _model = None
    return _model


def _build_text(site: dict | None, request_data) -> str:
    """Construye el texto de entrada para el clasificador."""
    parts: list[str] = []
    # Texto opcional que pudiera venir en el request (plan original).
    if request_data is not None:
        extra = getattr(request_data, "historicalInfo", None)
        if extra:
            parts.append(str(extra))
    # Texto del registro del sitio (fuente principal en el flujo actual).
    if site:
        parts.append(site.get("siteName", ""))
        parts.append(site.get("siteId", ""))
        parts.append(site.get("historicalInfo", ""))
        parts.append(" ".join(site.get("visualTags", [])))
    return " ".join(p for p in parts if p).strip()


def predict_reconstruction(site: dict | None, request_data=None) -> dict:
    """
    Devuelve { "class", "confidence", "modelVersion" }.
    Usa el modelo ML si está disponible; si no, fallback por categoría.
    """
    model = _get_model()
    text = _build_text(site, request_data)

    if model is not None and text:
        try:
            proba = model.predict_proba([text])[0]
            classes = list(model.classes_)
            idx = int(proba.argmax())
            return {
                "class": classes[idx],
                "confidence": round(float(proba[idx]), 4),
                "modelVersion": MODEL_VERSION,
            }
        except Exception as exc:
            print(f"[ml_service] Predicción falló, usando fallback: {exc}")

    # ── Fallback basado en la categoría del dataset ───────────────────────────
    fallback_class = (site or {}).get("category", "urban_historical")
    return {
        "class": fallback_class,
        "confidence": 0.75,
        "modelVersion": "fallback-rules-v1",
    }
