"""Reasoning engine over any API compatible with OpenAI's chat
completions schema: OpenAI, OpenRouter, Ollama, vLLM, LM Studio, Together,
Groq and so on. It is deliberately vendor-agnostic so the piece stays
reusable by anyone.

Note: `response_format: {"type": "json_object"}` is not supported the same
way by every OpenAI-compatible server (Ollama, for instance, uses
`"format": "json"` at the top level of the payload rather than inside
`response_format`). If your server does not support this field, override
`_extra_payload()` in a subclass to adapt it.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from ..types import Question
from .base import BaseEngine, HttpClientOwner

_PROMPT_TEMPLATE = """You are a decision engine. Reply with ONE valid JSON object and nothing else: no prose, no code fences.

Context:
{state}

Question ({qtype}): {instructions}
{criteria_block}

Required response format (the JSON only, nothing else):
{schema}
"""


def _schema_and_criteria(question: Question) -> tuple[str, str]:
    if question.type == "choice":
        opts = list(question.criteria.keys())  # type: ignore[union-attr]
        criteria_block = "Valid options: " + ", ".join(opts)
        schema = '{"choice": "<one of the options>", "confidence": <float 0-1>}'
    elif question.type == "score":
        levels = list(question.criteria.keys())  # type: ignore[union-attr]
        criteria_block = "Valid levels: " + ", ".join(str(lv) for lv in levels)
        schema = '{"score": "<one of the levels>", "confidence": <float 0-1>}'
    else:  # noul
        criteria_block = ""
        schema = '{"value": <true|false>, "confidence": <float 0-1>}'
    return schema, criteria_block


def _build_prompt(question: Question) -> str:
    schema, criteria_block = _schema_and_criteria(question)
    return _PROMPT_TEMPLATE.format(
        state=question.state or "(no additional context)",
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
    """A 'System 2' engine: slow, expensive, general-purpose. Sits at the
    end of the cascade by default (high relative_cost)."""

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
        # Only close the client if we were the ones who created it.
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
