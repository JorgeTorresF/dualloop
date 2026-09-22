"""Client for a self-hosted simple-jev server.

simple-jev (https://github.com/featherless-ai/simple-jev) turns any open
HF-Transformers-compatible model into a structured classification
endpoint, reading next-token logits for each predefined option. This
engine speaks ONLY the server's public HTTP protocol
(`POST /v1/classifier`); it embeds and redistributes none of simple-jev's
code, so this package's licence (MIT) is independent of the licence of
whatever server you point it at. Verify the licence of your own simple-jev
deployment separately: at the time of this research its repository had an
open issue about the absence of an explicit LICENSE file.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from ..types import Question
from .base import BaseEngine, HttpClientOwner


#: The key under which the question is sent to the jev server and read
#: back. It is an internal protocol identifier, not a domain label: this
#: used to be the `task_type`, which coupled your task naming to whatever
#: the server accepts as a key (and broke on a task_type carrying
#: unexpected characters). The same constant is sent and read, so it is
#: self-consistent.
_QUESTION_KEY = "decision"


class JevEngine(HttpClientOwner, BaseEngine):
    """A 'System 1' engine: fast, cheap, typed. Meant to go first in the
    arbitration cascade (low relative_cost)."""

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
        # Only close the client if we were the ones who created it.
        self._owns_client = client is None

    def _decide_raw(self, task_type: str, question: Question) -> tuple[Any, float, dict]:
        payload: dict[str, Any] = {
            "model": self.model,
            "state": question.state,
            "questions": {
                _QUESTION_KEY: {
                    "type": question.type,
                    "instructions": question.instructions,
                    **({"criteria": question.criteria} if question.criteria else {}),
                }
            },
        }
        resp = self._client.post(f"{self.base_url}/v1/classifier", json=payload)
        resp.raise_for_status()
        data = resp.json()
        answer = data["answers"][_QUESTION_KEY]

        if answer["type"] == "choice":
            return answer["choice"], float(answer["confidence"]), data
        if answer["type"] == "score":
            return answer["score"], float(answer["confidence"]), data
        if answer["type"] == "noul":
            # NOTE: the exact shape of a "noul" response was not
            # documented with an explicit example in the public README
            # consulted (only "choice" was). This parsing is the most
            # reasonable inference from the specification ("true/false
            # judgment, 0.01-0.99"). Check it against your own server's
            # /docs before relying on it in production, and adjust if the
            # real field is not called "value"/"score".
            p = float(answer.get("value", answer.get("score", 0.5)))
            return (p >= 0.5), max(p, 1 - p), data

        raise ValueError(f"Unrecognised Jev response type: {answer['type']!r}")
