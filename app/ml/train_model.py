"""
train_model.py
Entrena un clasificador REAL TF-IDF + Logistic Regression con los
`trainingSamples` del dataset y guarda el pipeline en
backend/app/ml/artifacts/reconstruction_classifier.joblib.

Es autocontenido (no importa el paquete `app`) para poder ejecutarse como:
    cd backend
    python app/ml/train_model.py
"""

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

APP_DIR = Path(__file__).resolve().parents[1]  # backend/app
DATASET_PATH = APP_DIR / "data" / "sites_dataset.json"
ARTIFACTS_DIR = APP_DIR / "ml" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "reconstruction_classifier.joblib"


def build_text(sample: dict, site_lookup: dict) -> str:
    """Combina inputText + visualTags + siteId (+ siteName) en un solo texto."""
    parts = [
        sample.get("inputText", ""),
        " ".join(sample.get("visualTags", [])),
        sample.get("siteId", ""),
    ]
    site = site_lookup.get(sample.get("siteId"))
    if site:
        parts.append(site.get("siteName", ""))
    return " ".join(p for p in parts if p).strip()


def main() -> None:
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("trainingSamples", [])
    if not samples:
        raise SystemExit("[train] No hay 'trainingSamples' en el dataset.")

    site_lookup = {s["siteId"]: s for s in data.get("sites", [])}
    X = [build_text(s, site_lookup) for s in samples]
    y = [s["label"] for s in samples]

    # Hiperparámetros: 3 clases bien separadas y vocabulario corto en español.
    # sublinear_tf + C alto dan probabilidades más nítidas (confianza realista),
    # sin dejar de ser un modelo real que puede equivocarse con texto ambiguo.
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True, sublinear_tf=True)),
            ("clf", LogisticRegression(C=10.0, max_iter=2000)),
        ]
    )
    pipeline.fit(X, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    classes = sorted(set(y))
    accuracy = pipeline.score(X, y)
    print(f"[train] Muestras entrenadas: {len(samples)}")
    print(f"[train] Clases detectadas ({len(classes)}): {', '.join(classes)}")
    print(f"[train] Accuracy en entrenamiento: {accuracy:.2f}")
    print(f"[train] Modelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
