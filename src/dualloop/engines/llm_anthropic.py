"""Reasoning engine over Anthropic's native Messages API.

Offered as its own adapter, rather than forcing Anthropic through the
OpenAI schema, because its API is not compatible with OpenAI chat
completions. It shares the same prompt and schema as `llm_openai` so both
engines are interchangeable from the Arbiter's point of view.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from ..types import Question
from .base import BaseEngine, HttpClientOwner
from .llm_openai import _build_prompt, _parse_llm_json


class AnthropicLLMEngine(HttpClientOwner, BaseEngine):
    """A 'System 2' engine over Claude. High relative_cost by default: it
    sits at the end of the cascade unless the bandit learns otherwise."""

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
