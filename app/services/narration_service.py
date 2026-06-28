"""
narration_service.py
Genera una narración histórica dinámica usando Groq (LLaMA 3) con streaming.

Recibe los metadatos del sitio y devuelve un generador de chunks de texto.

Protecciones:
  - Si la API key no está configurada → fallback estático silencioso.
  - Rate limiting en memoria (sliding window por IP/global) → fallback silencioso.
  - Si Groq falla por cualquier razón → fallback estático silencioso.
  En ningún caso el usuario ve un error; siempre recibe texto.
"""

import os
import time
from collections import deque
from collections.abc import Generator
from threading import Lock

from groq import Groq

# ── Cliente Groq (inicialización perezosa) ────────────────────────────────────
_client: Groq | None = None


def _get_client() -> Groq | None:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        try:
            _client = Groq(api_key=api_key)
        except Exception as exc:
            print(f"[narration_service] No se pudo crear el cliente Groq: {exc}")
    return _client


# ── Rate limiter: sliding window en memoria ───────────────────────────────────
# Groq free tier: ~30 req/min. Usamos un límite conservador de 10 req/min
# para dejar margen a otros endpoints y evitar errores 429.
_RATE_LIMIT   = int(os.getenv("NARRATE_RATE_LIMIT", "10"))   # máx requests
_RATE_WINDOW  = int(os.getenv("NARRATE_RATE_WINDOW", "60"))  # en segundos

_request_times: deque = deque()   # timestamps de requests recientes
_rate_lock = Lock()


def _is_rate_limited() -> bool:
    """Devuelve True si se superó el límite de requests en la ventana."""
    now = time.monotonic()
    with _rate_lock:
        # Elimina timestamps fuera de la ventana deslizante.
        while _request_times and now - _request_times[0] > _RATE_WINDOW:
            _request_times.popleft()

        if len(_request_times) >= _RATE_LIMIT:
            return True

        _request_times.append(now)
        return False


# ── Narraciones de fallback (una por sitio) ───────────────────────────────────
_FALLBACK: dict[str, str] = {
    "centro_lima": (
        "Ante ti se extiende el Centro Histórico de Lima en el año 2010, "
        "rescatado del olvido y restaurado a su esplendor colonial. "
        "Los balcones de madera tallada y los faroles encendidos te recuerdan "
        "que aquí latió el corazón del Virreinato del Perú desde 1535. "
        "Explora los puntos de interés para descubrir los secretos de la Ciudad de los Reyes."
    ),
    "chan_chan": (
        "Ante ti se extiende Chan Chan en 1450, en el apogeo del reino Chimú. "
        "Los muros de adobe se elevan cubiertos de relieves geométricos que narran "
        "siglos de historia de la ciudad de barro más grande del mundo. "
        "Acércate a las plazas y relieves para sentir la grandeza de esta civilización."
    ),
    "real_felipe": (
        "Ante ti se alza la Fortaleza del Real Felipe en 1800, centinela de piedra "
        "del puerto del Callao. Los cañones de bronce apuntan al horizonte marino, "
        "listos para defender el paso más estratégico del Virreinato. "
        "Explora las torres y el patio de armas para conocer su historia militar."
    ),
}


def _fallback_narration(site: dict, target_year: int) -> str:
    """Texto estático por sitio; genérico si el sitio no tiene uno específico."""
    site_id = site.get("siteId", "")
    if site_id in _FALLBACK:
        return _FALLBACK[site_id]
    name = site.get("siteName", "este sitio")
    return (
        f"Ante ti se extiende {name} en el año {target_year}, "
        f"restaurado a su máximo esplendor. "
        f"Los vestigios del tiempo han desaparecido y puedes contemplar "
        f"su arquitectura original en todo su detalle. "
        f"Explora los puntos de interés para descubrir su historia."
    )


# ── Prompt ────────────────────────────────────────────────────────────────────
def _build_prompt(site: dict, target_year: int) -> str:
    name              = site.get("siteName", "el sitio")
    historical_info   = site.get("historicalInfo", "")
    tags              = ", ".join(site.get("visualTags", []))
    reconstructed     = ", ".join(site.get("reconstructedCues", []))

    return (
        f"Eres el narrador del dispositivo Chrono-Vision, una herramienta de "
        f"exploración temporal del patrimonio histórico del Perú.\n\n"
        f"El usuario acaba de activar el dispositivo y está viendo la reconstrucción "
        f"de \"{name}\" en el año {target_year}.\n\n"
        f"Información histórica: {historical_info}\n"
        f"Elementos reconstruidos: {reconstructed}\n"
        f"Etiquetas visuales: {tags}\n\n"
        f"Genera una narración histórica inmersiva de exactamente 3 oraciones completas "
        f"(80-100 palabras).\n"
        f"Requisitos:\n"
        f"- Segunda persona (\"Ante ti...\", \"Puedes ver...\")\n"
        f"- Año {target_year} como contexto temporal\n"
        f"- 1 detalle histórico concreto\n"
        f"- Tono épico pero educativo\n"
        f"- Última oración invita a explorar los puntos de interés\n"
        f"- Termina con punto final\n\n"
        f"Responde SOLO con la narración, sin título ni comentarios."
    )


# ── Stream principal ──────────────────────────────────────────────────────────
def stream_narration(site: dict, target_year: int) -> Generator[str, None, None]:
    """
    Genera la narración como stream de chunks.
    Nunca lanza — siempre devuelve algo (Groq real o fallback).
    """
    # 1) Rate limit → fallback silencioso
    if _is_rate_limited():
        print("[narration_service] Rate limit alcanzado, usando fallback.")
        yield _fallback_narration(site, target_year)
        return

    # 2) Sin cliente → fallback silencioso
    client = _get_client()
    if client is None:
        yield _fallback_narration(site, target_year)
        return

    # 3) Llamada a Groq con streaming
    try:
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": _build_prompt(site, target_year)}],
            max_tokens=350,
            temperature=0.75,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception as exc:
        print(f"[narration_service] Groq error: {exc}")
        yield _fallback_narration(site, target_year)
