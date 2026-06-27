"""
narration_service.py
Genera una narración histórica dinámica usando Groq (LLaMA 3) con streaming.

Recibe los metadatos del sitio y devuelve un generador de chunks de texto.
Si la API key no está configurada o Groq falla, devuelve un texto de fallback
estático para no romper la experiencia.
"""

import os
from collections.abc import Generator

from groq import Groq

_client: Groq | None = None


def _get_client() -> Groq | None:
    """Inicialización perezosa del cliente Groq."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        _client = Groq(api_key=api_key)
    return _client


def _build_prompt(site: dict, target_year: int) -> str:
    """Construye el prompt que le enviamos a la LLM."""
    name = site.get("siteName", "el sitio")
    historical_info = site.get("historicalInfo", "")
    tags = ", ".join(site.get("visualTags", []))
    reconstructed_cues = ", ".join(site.get("reconstructedCues", []))

    return f"""Eres el narrador del dispositivo Chrono-Vision, una herramienta de exploración temporal del patrimonio histórico del Perú.

El usuario acaba de activar el dispositivo y está viendo la reconstrucción de "{name}" en el año {target_year}.

Información histórica del sitio: {historical_info}
Elementos visuales reconstruidos: {reconstructed_cues}
Etiquetas visuales del sitio: {tags}

Genera una narración histórica inmersiva y emocionante de exactamente 3 oraciones completas (80-100 palabras).
Debe:
- Estar escrita en segunda persona ("Ante ti se extiende...", "Puedes ver...")
- Describir lo que el explorador observa en el año {target_year}
- Mencionar 1 detalle histórico concreto del sitio
- Tener un tono épico pero educativo
- La última oración debe invitar a explorar los puntos de interés

IMPORTANTE: Escribe exactamente 3 oraciones. Termina siempre con punto final.
Responde SOLO con la narración, sin título ni comentarios adicionales."""


def _fallback_narration(site: dict, target_year: int) -> str:
    """Texto estático si Groq no está disponible."""
    name = site.get("siteName", "este sitio")
    return (
        f"Ante ti se extiende {name} en el año {target_year}, "
        f"restaurado a su máximo esplendor. Los vestigios del tiempo han desaparecido "
        f"y puedes contemplar su arquitectura original en todo su detalle. "
        f"Explora los puntos de interés para descubrir su historia."
    )


def stream_narration(site: dict, target_year: int) -> Generator[str, None, None]:
    """
    Genera la narración histórica como stream de chunks de texto.

    Yields:
        str: fragmentos de texto según llegan de Groq.
    """
    client = _get_client()

    if client is None:
        # Sin API key: devuelve el fallback como un único chunk.
        yield _fallback_narration(site, target_year)
        return

    try:
        prompt = _build_prompt(site, target_year)

        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # rápido y gratuito en Groq
            messages=[{"role": "user", "content": prompt}],
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
