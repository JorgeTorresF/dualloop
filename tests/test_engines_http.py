"""Lifecycle of the engines' HTTP client.

An engine closes the client it created, and never the one it was handed:
closing someone else's client would break any other engine sharing it.
"""

import httpx

from dualloop.engines import AnthropicLLMEngine, JevEngine, OpenAICompatibleLLMEngine

ENGINES = [
    lambda **kw: JevEngine(base_url="http://x", model="m", **kw),
    lambda **kw: OpenAICompatibleLLMEngine(base_url="http://x", api_key="k", model="m", **kw),
    lambda **kw: AnthropicLLMEngine(api_key="k", model="m", **kw),
]


def test_closes_the_client_it_created():
    for build in ENGINES:
        engine = build()
        assert engine._owns_client is True
        engine.close()
        assert engine._client.is_closed, f"{engine.name} did not close its own client"


def test_does_not_close_an_injected_client():
    for build in ENGINES:
        injected = httpx.Client()
        engine = build(client=injected)
        assert engine._owns_client is False
        engine.close()
        assert not injected.is_closed, f"{engine.name} closed a client that was not its own"
        injected.close()


def test_close_is_idempotent():
    for build in ENGINES:
        engine = build()
        engine.close()
        engine.close()  # must not raise


def test_works_as_a_context_manager():
    for build in ENGINES:
        with build() as engine:
            assert not engine._client.is_closed
        assert engine._client.is_closed, f"{engine.name} did not close on leaving the with block"
