# Chrono-Vision — Backend (FastAPI + ML)

Backend de reconstrucción para **Chrono-Vision**. Recibe una solicitud del
frontend, consulta el dataset, ejecuta un **clasificador ML real**
(TF-IDF + Logistic Regression) y devuelve un **JSON compatible con el
renderizador A-Frame** existente (no hay que tocar el motor de escena).

> No incluye Firebase, autenticación, subida de imágenes ni generación con IA.
> Firebase (en el frontend) se usa aparte, solo para guardar interacciones.

## Arquitectura

```
backend/
  app/
    main.py                      # FastAPI + CORS + handler de errores
    config.py                    # rutas, versión, CORS, mapeo clase→plantilla
    api/routes.py                # endpoints
    schemas/reconstruction.py    # modelos Pydantic (request/response)
    services/
      dataset_service.py         # lee sites_dataset.json
      ml_service.py              # carga el modelo y predice (con fallback)
      reconstruction_service.py  # arma el JSON final desde la plantilla
    data/sites_dataset.json      # sitios + trainingSamples (copiado del frontend)
    templates/                   # escenas verificadas (env/objects/hotspots/effects)
      urban_historical.json
      archaeological_zone.json
      coastal_fortress.json
    ml/
      train_model.py             # entrena y guarda el modelo
      artifacts/                 # reconstruction_classifier.joblib (generado)
  requirements.txt
```

## Instalación

```bash
cd backend            # (carpeta chrono-vision-backend)
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux/Mac
pip install -r requirements.txt
```

## Entrenar el modelo ML

```bash
python app/ml/train_model.py
```
Lee `app/data/sites_dataset.json`, entrena TF-IDF + Logistic Regression con los
`trainingSamples` y guarda `app/ml/artifacts/reconstruction_classifier.joblib`.
Imprime cuántas muestras entrenó y qué clases detectó.

> La API funciona aunque el modelo no esté entrenado: usa un **fallback** por
> categoría del sitio.

## Ejecutar

```bash
uvicorn app.main:app --reload --port 8000
```
- API: http://localhost:8000
- Swagger: http://localhost:8000/docs

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Estado del servicio |
| GET | `/sites` | Lista resumida de sitios |
| GET | `/sites/{site_id}` | Info completa del sitio (404 si no existe) |
| POST | `/reconstruct` | Predicción ML + JSON de reconstrucción |

### Ejemplo `POST /reconstruct`

```bash
curl -X POST http://localhost:8000/reconstruct ^
  -H "Content-Type: application/json" ^
  -d "{\"siteId\":\"centro_lima\",\"mlCategory\":\"urban_historical\",\"currentYear\":2077,\"targetYear\":2010}"
```

Respuesta (resumida):
```json
{
  "siteId": "centro_lima",
  "sceneName": "Centro Histórico de Lima — Reconstrucción 2010",
  "currentYear": 2077,
  "targetYear": 2010,
  "prediction": { "class": "urban_historical", "confidence": 0.93, "modelVersion": "tfidf-logreg-v1" },
  "environment": { "...": "..." },
  "objects": [ "..." ],
  "hotspots": [ "..." ],
  "effects": { "...": "..." },
  "source": { "datasetVersion": "1.0.0", "templateUsed": "urban_historical.json" }
}
```

> El `request` real del frontend es `{ siteId, mlCategory, currentYear, targetYear }`.
> Solo `siteId` es obligatorio; el resto es opcional.

## Conectar con el frontend

En el frontend (`chrono-vision/.env.local`):
```bash
VITE_USE_MOCK_API=false
VITE_API_BASE_URL=http://localhost:8000
```
Reinicia `npm run dev`. El selector, la transición y los hotspots funcionan
igual, pero ahora la reconstrucción viene del backend real.

## Qué es ML real y qué es plantilla

- **ML real:** `prediction.class` y `prediction.confidence` salen del modelo
  TF-IDF + Logistic Regression entrenado con el dataset. Sin modelo entrenado,
  cae a un fallback por categoría.
- **Plantillas:** `environment`, `objects`, `hotspots` y `effects` provienen de
  las plantillas de escena (copia 1:1 de la versión ya verificada en el
  frontend). El JSON 3D **no** lo genera el ML; el ML elige qué plantilla usar.

## CORS

Permitidos por defecto: `http://localhost:5173` y `http://127.0.0.1:5173`.
Para agregar otro origen (deploy), definir `FRONTEND_ORIGIN` en `.env`
(ver `.env.example`).
