"""Motor de razonamiento vía la API nativa de Anthropic (Messages API).

Se ofrece como adaptador propio (en vez de forzar Anthropic a través del
esquema OpenAI) porque su API no es compatible con chat completions de
OpenAI. Comparte el mismo prompt/esquema que `llm_openai` para que ambos
motores sean intercambiables desde el punto de vista del Arbiter.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from ..types import Question
from .base import BaseEngine, HttpClientOwner
from .llm_openai import _build_prompt, _parse_llm_json


class AnthropicLLMEngine(HttpClientOwner, BaseEngine):
    """Motor 'System 2' vía Claude. relative_cost alto por defecto: va al
    final de la cascada salvo que el bandit aprenda lo contrario."""

    relative_cost = 5.0

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        name: str = "llm-anthropic",
        timeout: float = 30.0,
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 300,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        self._client = client or httpx.Client(timeout=timeout, headers=headers)
        # Solo cerramos el cliente si lo hemos creado nosotros.
        self._owns_client = client is None

    def _decide_raw(self, task_type: str, question: Question) -> tuple[Any, float, dict]:
        prompt = _build_prompt(question)
        resp = self._client.post(
            f"{self.base_url}/v1/messages",
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["content"][0]["text"]
        parsed = _parse_llm_json(content)
        confidence = float(parsed.get("confidence", 0.5))

        if question.type == "choice":
            return parsed["choice"], confidence, data
        if question.type == "score":
            return parsed["score"], confidence, data
        return bool(parsed["value"]), confidence, data
