"""Ciclo de vida del cliente HTTP de los motores.

Un motor cierra el cliente que creo el, y nunca el que le inyectaron:
cerrar un cliente ajeno romperia a cualquier otro motor que lo comparta.
"""

import httpx

from dualloop.engines import AnthropicLLMEngine, JevEngine, OpenAICompatibleLLMEngine

MOTORES = [
    lambda **kw: JevEngine(base_url="http://x", model="m", **kw),
    lambda **kw: OpenAICompatibleLLMEngine(base_url="http://x", api_key="k", model="m", **kw),
    lambda **kw: AnthropicLLMEngine(api_key="k", model="m", **kw),
]


def test_cierra_el_cliente_que_creo_el():
    for build in MOTORES:
        engine = build()
        assert engine._owns_client is True
        engine.close()
        assert engine._client.is_closed, f"{engine.name} no cerro su propio cliente"


def test_no_cierra_un_cliente_inyectado():
    for build in MOTORES:
        inyectado = httpx.Client()
        engine = build(client=inyectado)
        assert engine._owns_client is False
        engine.close()
        assert not inyectado.is_closed, f"{engine.name} cerro un cliente que no era suyo"
        inyectado.close()


def test_close_es_idempotente():
    for build in MOTORES:
        engine = build()
        engine.close()
        engine.close()  # no debe lanzar


def test_sirve_como_gestor_de_contexto():
    for build in MOTORES:
        with build() as engine:
            assert not engine._client.is_closed
        assert engine._client.is_closed, f"{engine.name} no cerro al salir del with"
