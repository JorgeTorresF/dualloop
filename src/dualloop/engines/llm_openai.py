"""Motor de razonamiento vía cualquier API compatible con el esquema de
chat completions de OpenAI: OpenAI, OpenRouter, Ollama, vLLM, LM Studio,
Together, Groq, etc. Es deliberadamente genérico (no atado a un vendor)
para que la pieza sea reutilizable por cualquiera, tal como se decidió al
construir esta librería.

Nota: `response_format: {"type": "json_object"}` no lo soportan todos los
servidores OpenAI-compatibles de la misma forma (p.ej. Ollama usa
`"format": "json"` en el nivel superior del payload, no dentro de
`response_format`). Si tu servidor no soporta este campo, sobrescribe
`_extra_payload()` en una subclase para adaptarlo.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from ..types import Question
from .base import BaseEngine, HttpClientOwner

_PROMPT_TEMPLATE = """Eres un motor de decisión. Responde UNICAMENTE con un objeto JSON valido, sin texto adicional ni bloques de codigo.

Contexto:
{state}

Pregunta ({qtype}): {instructions}
{criteria_block}

Formato de respuesta requerido (solo el JSON, nada mas):
{schema}
"""


def _schema_and_criteria(question: Question) -> tuple[str, str]:
    if question.type == "choice":
        opts = list(question.criteria.keys())  # type: ignore[union-attr]
        criteria_block = "Opciones validas: " + ", ".join(opts)
        schema = '{"choice": "<una de las opciones>", "confidence": <float 0-1>}'
    elif question.type == "score":
        levels = list(question.criteria.keys())  # type: ignore[union-attr]
        criteria_block = "Niveles validos: " + ", ".join(str(lv) for lv in levels)
        schema = '{"score": "<uno de los niveles>", "confidence": <float 0-1>}'
    else:  # noul
        criteria_block = ""
        schema = '{"value": <true|false>, "confidence": <float 0-1>}'
    return schema, criteria_block


def _build_prompt(question: Question) -> str:
    schema, criteria_block = _schema_and_criteria(question)
    return _PROMPT_TEMPLATE.format(
        state=question.state or "(sin contexto adicional)",
        qtype=question.type,
        instructions=question.instructions,
        criteria_block=criteria_block,
        schema=schema,
    )


def _parse_llm_json(content: str) -> dict:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(content[start : end + 1])


class OpenAICompatibleLLMEngine(HttpClientOwner, BaseEngine):
    """Motor 'System 2': lento, caro, generalista. Va al final de la
    cascada por defecto (relative_cost alto)."""

    relative_cost = 5.0

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        name: str = "llm",
        timeout: float = 30.0,
        extra_body: Optional[dict[str, Any]] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.extra_body = extra_body or {"response_format": {"type": "json_object"}}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout, headers=headers)
        # Solo cerramos el cliente si lo hemos creado nosotros.
        self._owns_client = client is None

    def _decide_raw(self, task_type: str, question: Question) -> tuple[Any, float, dict]:
        prompt = _build_prompt(question)
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            **self.extra_body,
        }
        resp = self._client.post(f"{self.base_url}/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = _parse_llm_json(content)
        confidence = float(parsed.get("confidence", 0.5))

        if question.type == "choice":
            return parsed["choice"], confidence, data
        if question.type == "score":
            return parsed["score"], confidence, data
        return bool(parsed["value"]), confidence, data
