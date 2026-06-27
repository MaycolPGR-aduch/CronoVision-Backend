"""
main.py
Punto de entrada FastAPI de Chrono-Vision. Configura CORS, registra las rutas y
un handler genérico para no exponer stack traces al usuario final.

Ejecutar (desde backend/):  uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import CORS_ORIGINS, SERVICE_NAME, SERVICE_VERSION

app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description="API de reconstrucción de Chrono-Vision (clasificación ML + plantillas de escena A-Frame).",
)

# CORS: permite al frontend Vite (5173) consumir la API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Evita filtrar stack traces; las HTTPException siguen su curso normal."""
    print(f"[main] Error no controlado en {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Error interno del servidor."})
