# DualLoop

Arbitraje aprendido y auditable entre motores de decisión heterogéneos
(un LLM de razonamiento, un clasificador tipado tipo [Jev/simple-jev](https://github.com/featherless-ai/simple-jev),
reglas deterministas...), con **cierre del bucle decisión → resultado →
recalibración sin reentrenamiento manual**.

## Por qué existe esto

Es habitual combinar hoy un LLM "System 2" (razonamiento lento, caro,
general) con un clasificador rápido y tipado "System 1" — el propio
[TypeSafe AI/Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
se lanzó en septiembre de 2026 posicionándose explícitamente así. Lo que
**no existe todavía, ni en frameworks de orquestación mainstream
(LangGraph, CrewAI, AutoGen, Semantic Kernel) ni en los routers
comerciales de LLM** (RouteLLM, OpenRouter, Martian, Not Diamond — todos
enrutan solo entre variantes de LLM, nunca entre motores de naturaleza
distinta) son dos piezas concretas:

1. **Un árbitro aprendido y auditable entre motores heterogéneos.** Ningún
   trabajo revisado arbitra entre un LLM, un clasificador tipado y reglas
   deterministas bajo un contrato común. Los meta-controladores existentes
   rutan entre **variantes de LLM** — AAMC
   ([ScienceDirect S0925231226005898](https://www.sciencedirect.com/science/article/pii/S0925231226005898))
   orquesta SLM/LLM con aprendizaje por refuerzo — o ni siquiera eso:
   Meta-Reasoner ([arXiv:2502.19918](https://arxiv.org/abs/2502.19918))
   opera *dentro* de un único modelo, usando bandits contextuales para
   decidir cuándo retroceder, cambiar de enfoque o reiniciar el
   razonamiento. Es el pariente más cercano de DualLoop en el uso de
   bandits para meta-decisiones, pero su espacio de acciones son
   estrategias de razonamiento, no motores.
2. **Cierre del bucle sin reentrenamiento manual.** La investigación sobre
   memoria de agentes (Mem0, ["State of AI Agent Memory 2026"](https://mem0.ai/blog/state-of-ai-agent-memory-2026))
   confirma que la memoria *procedimental* (aprender de outcomes) "sigue en
   etapa temprana"; los frameworks de "decision provenance"
   ([arXiv:2602.22442](https://arxiv.org/html/2602.22442v1)) se detienen
   deliberadamente en auditoría para humanos y no llegan al auto-ajuste; y
   el trabajo más cercano a una solución sin gradientes (JitRL,
   [arXiv:2601.18510](https://arxiv.org/abs/2601.18510)) sigue siendo un
   prototipo de investigación, no una librería usable.

DualLoop no resuelve esto con un modelo nuevo, sino con matemática simple
y auditable (calibración bayesiana por bins + bandits Beta-Bernoulli +
aproximación estocástica de paso constante) que funciona con **cualquier
motor, incluidos APIs cerradas de caja negra** — no requiere acceso a
logits ni a los pesos del modelo, a diferencia de JitRL. El detalle
completo de las decisiones de diseño y sus fuentes está en
[`docs/architecture.md`](docs/architecture.md).

## Instalación

```bash
pip install "dualloop @ git+https://github.com/JorgeTorresF/dualloop.git"
```

O en modo desarrollo, clonando el repo:

```bash
git clone https://github.com/JorgeTorresF/dualloop.git
cd dualloop
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Sin dependencias obligatorias más allá de `httpx`. El extra `demo` añade
`matplotlib` (opcional, solo para graficar el ejemplo sintético).

## Quickstart

```python
from dualloop import DualLoop, Question
from dualloop.engines import JevEngine, OpenAICompatibleLLMEngine, RuleEngine

engines = [
    JevEngine(base_url="http://127.0.0.1:8000", model="Qwen/Qwen3.5-0.8B"),
    OpenAICompatibleLLMEngine(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-...",
        model="anthropic/claude-sonnet-5",
    ),
    # Una regla determinista gana a cualquier modelo cuando el caso es obvio:
    # si hay posología explícita, es una prescripción y no hay nada que deliberar.
    RuleEngine(lambda task_type, q: ("prescribe", 0.99) if " mg" in q.state else None),
]

loop = DualLoop(engines=engines)

# Gate de estilo sobre un informe que va a leer un paciente: el informe
# recomienda y deriva al médico, nunca pauta.
question = Question(
    type="choice",
    instructions="¿Este párrafo prescribe un tratamiento o solo recomienda?",
    state="Conviene que valores con tu médico si procede revisar la pauta actual.",
    criteria={"recomienda": None, "prescribe": None},
)

decision = loop.decide(task_type="gate_prescripcion", question=question)
print(decision.chosen_engine, decision.chosen_value, decision.chosen_confidence)

# ... cuando el revisor humano confirma o corrige el veredicto ...
loop.report_outcome(decision.id, correct=True)
```

**Por qué este ejemplo y no otro.** DualLoop aprende de outcomes, así que
solo rinde donde los outcomes llegan pronto y en cantidad: decenas o
cientos de casos por tipo de tarea, con la verdad conocida en horas o días.
Un gate de revisión sobre documentos que ya se revisan a diario encaja. Una
decisión que se toma cinco veces al año y cuya verdad tarda meses —evaluar
una solicitud de subvención, por ejemplo— no encaja: el calibrador nunca
sale del arranque en frío y no notarás diferencia frente a una regla fija.

Cada `report_outcome()` recalibra online la confianza de los motores para
ese tipo de tarea, ajusta qué tan fiable parece cada motor, y mueve el
umbral de aceptación — sin tocar ningún peso de modelo ni reentrenar nada.

Para Anthropic/Claude en vez de un endpoint OpenAI-compatible:

```python
from dualloop.engines import AnthropicLLMEngine

llm = AnthropicLLMEngine(api_key="sk-ant-...", model="claude-sonnet-5")
```

Para persistir el estado entre reinicios del proceso:

```python
from dualloop import SQLiteStore

loop = DualLoop(engines=engines, store=SQLiteStore("dualloop.db"))
```

## Cómo funciona (resumen)

1. **`decide()`** ordena los motores disponibles por fiabilidad aprendida
   (bandit Thompson sampling), consulta el más prometedor primero, calibra
   su confianza cruda y, si supera el umbral adaptativo del tipo de tarea,
   acepta esa respuesta sin seguir escalando. Si no, prueba el siguiente
   motor. Todo queda registrado en un `Decision` auditable.
2. **`report_outcome()`** actualiza, con una sola llamada, tres
   estadísticos online: la calibración de confianza del motor usado, su
   fiabilidad agregada para ese tipo de tarea, y el umbral de aceptación —
   todo con actualizaciones bayesianas cerradas en O(1), sin gradientes.
3. **`explain()`** reconstruye por qué se tomó una decisión: qué motores
   se consultaron, sus votos crudos y calibrados, el umbral usado, y la
   fiabilidad aprendida en ese momento.

Ver [`docs/architecture.md`](docs/architecture.md) para el detalle
matemático y la justificación de cada elección de diseño.

## Abstención: cuando el sistema no sabe

Por defecto, si ningún motor alcanza el umbral, el loop acepta igualmente el
voto de mayor confianza calibrada: el umbral queda registrado en la decisión
pero no actúa como suelo. Para un gate donde «no sé, que lo mire una
persona» es una respuesta legítima, eso no vale:

```python
loop = DualLoop(engines=engines, abstain_below_threshold=True)

decision = loop.decide(task_type="gate_prescripcion", question=question)
if decision.abstained:
    enviar_a_revision_humana(decision)   # chosen_value sigue disponible
else:
    aplicar(decision.chosen_value)
```

Una decisión abstenida **no mueve el umbral de aceptación** cuando reportas
su outcome: el umbral persigue la tasa de error de lo *aceptado*, y una
abstención no se aceptó — endurecerlo por un caso que el propio sistema ya
había marcado como dudoso rompería su semántica. Los calibradores y el
bandit sí aprenden de ella.

## Demo sintética

```bash
python examples/demo_synthetic.py
```

Simula 800 decisiones con tres motores de fiabilidad y coste distintos
(uno de ellos deliberadamente sobreconfiado en casos difíciles) y muestra
cómo la precisión, el coste (motores consultados por decisión) y el error
de calibración evolucionan con la experiencia, sin ninguna intervención
manual entre medias.

## Limitaciones honestas

- La demo es **sintética**: prueba que el mecanismo de recalibración
  funciona como está diseñado, no que resuelve ningún dominio real. No ha
  sido validado en producción.
- El parseo de respuestas `noul` de `JevEngine` está inferido de la
  especificación pública de simple-jev, no confirmado contra un ejemplo
  de respuesta real — verifícalo contra el `/docs` de tu propio servidor
  antes de usarlo en producción.
- `report_outcome()` sin `per_engine_correct` aproxima el acierto de los
  motores no elegidos por si coincidieron con la respuesta elegida; para
  una recalibración más precisa, pasa la verdad conocida por motor cuando
  la tengas (por ejemplo, en un set de evaluación offline).
- La licencia del propio servidor `simple-jev` no estaba claramente
  especificada en su repositorio a fecha de esta investigación (hay un
  issue abierto al respecto) — este cliente solo habla su protocolo HTTP
  público y no redistribuye su código, pero verifica la licencia de tu
  propio despliegue por separado.
- El bucketing de contexto es por `task_type` explícito, no por similitud
  semántica entre casos (ver Roadmap).

## Roadmap / extensiones posibles

- Bucketing de contexto por similitud (embeddings) en vez de solo por
  `task_type`, para generalizar calibración entre casos parecidos.
- Backends de store adicionales (Postgres, Redis) implementando el mismo
  protocolo `Store`.
- Verificación formal como motor adicional para dominios con estructura
  lógica explícita (código, cumplimiento normativo).
- Heurísticos de inferencia de outcome adicionales, documentados con sus
  limitaciones de forma tan explícita como `HumanOverrideHeuristic`.

## Licencia

MIT — ver [`LICENSE`](LICENSE).
