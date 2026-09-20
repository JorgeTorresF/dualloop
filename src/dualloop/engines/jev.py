"""Cliente para un servidor simple-jev autoalojado.

simple-jev (https://github.com/featherless-ai/simple-jev) convierte
cualquier modelo abierto compatible con HF Transformers en un endpoint de
clasificación estructurada, leyendo los logits del siguiente token para
cada opción predefinida. Este motor SOLO habla el protocolo HTTP público
del servidor (`POST /v1/classifier`) — no embebe ni redistribuye código de
simple-jev, así que la licencia de este paquete (MIT) es independiente de
la del servidor que se esté consultando (verifica la licencia de tu propio
despliegue de simple-jev por separado; a fecha de esta investigación su
repo tenía un issue abierto sobre falta de fichero LICENSE explícito).
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from ..types import Question
from .base import BaseEngine


class JevEngine(BaseEngine):
    """Motor 'System 1': rápido, barato, tipado. Pensado para ir primero en
    la cascada de arbitraje (relative_cost bajo)."""

    relative_cost = 0.1

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        name: str = "jev",
        timeout: float = 5.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def _decide_raw(self, task_type: str, question: Question) -> tuple[Any, float, dict]:
        payload: dict[str, Any] = {
            "model": self.model,
            "state": question.state,
            "questions": {
                task_type: {
                    "type": question.type,
                    "instructions": question.instructions,
                    **({"criteria": question.criteria} if question.criteria else {}),
                }
            },
        }
        resp = self._client.post(f"{self.base_url}/v1/classifier", json=payload)
        resp.raise_for_status()
        data = resp.json()
        answer = data["answers"][task_type]

        if answer["type"] == "choice":
            return answer["choice"], float(answer["confidence"]), data
        if answer["type"] == "score":
            return answer["score"], float(answer["confidence"]), data
        if answer["type"] == "noul":
            # NOTA: el formato exacto de la respuesta "noul" no aparecía
            # documentado con un ejemplo explícito en el README público
            # consultado (solo el de "choice" lo estaba). Este parseo es la
            # mejor inferencia razonable a partir de la especificación
            # ("juicio verdadero/falso, 0.01-0.99") — verifica contra el
            # /docs de tu propio servidor antes de usar en producción y
            # ajusta si el campo real no se llama "value"/"score".
            p = float(answer.get("value", answer.get("score", 0.5)))
            return (p >= 0.5), max(p, 1 - p), data

        raise ValueError(f"Tipo de respuesta Jev no reconocido: {answer['type']!r}")
