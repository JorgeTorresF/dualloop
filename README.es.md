# DualLoop

[![CI](https://github.com/JorgeTorresF/dualloop/actions/workflows/ci.yml/badge.svg)](https://github.com/JorgeTorresF/dualloop/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*Read this README [in English](README.md) — that is the canonical version.*

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
[`docs/architecture.es.md`](docs/architecture.es.md).

## ¿Te sirve esto?

DualLoop no es un framework de agentes ni un router de LLMs. Es una pieza
pequeña para un problema concreto: **tienes varias formas de decidir lo
mismo, con coste y fiabilidad distintos, y quieres usar la barata cuando
basta y la cara cuando hace falta — sin fijar a mano dónde está ese
límite.**

Encaja si tu caso cumple las cuatro:

1. **La decisión se repite.** Decenas o cientos de casos del mismo tipo, no
   una deliberación única.
2. **La respuesta es tipada.** Una opción entre varias, una puntuación
   sobre una rúbrica, un juicio de sí/no. No prosa libre.
3. **La verdad acaba llegando.** Alguien revisa, el caso se resuelve, el
   cliente responde. En horas o días, no en meses.
4. **Tienes motores desiguales.** Un clasificador pequeño y una API cara;
   o reglas y un LLM. Si solo tienes un motor, no hay nada que arbitrar.

Ejemplos que cumplen las cuatro:

- **Gates de revisión de contenido.** Publicaciones, informes o respuestas
  que pasan por un filtro antes de salir; quien revisa confirma o corrige,
  y esa corrección es la verdad.
- **Triaje de tickets o incidencias.** Categoría y prioridad; la verdad es
  dónde acabó realmente el ticket.
- **Moderación y detección de abuso.** Reglas baratas para lo evidente,
  clasificador para el volumen, LLM para lo ambiguo.
- **Control de calidad de extracción de datos.** ¿Este campo extraído es
  correcto? Los muestreos humanos realimentan el sistema.
- **Gates en CI.** ¿Este cambio necesita revisión humana? La verdad es si
  quien revisó encontró algo.
- **Cualificación de leads o filtrado de spam**, donde el desenlace se
  conoce poco después.

Y casos que **no** encajan, para ahorrarte la decepción: decisiones que
tomas cinco veces al año; verdades que tardan meses en conocerse; salidas
en prosa; o un único motor. En todos ellos el calibrador se queda en el
arranque en frío y no notarás diferencia frente a un umbral fijo escrito a
mano.

**Lo que aporta frente a hacerlo tú.** Podrías escribir el `if
confianza > 0.8` a mano. Lo que no es trivial es lo demás: que la confianza
de motores distintos sea comparable entre sí, que ese `0.8` se mueva solo
según los errores que de verdad cometes, que el orden de consulta aprenda
cuál es fiable para cada tipo de tarea, y que todo quede auditable caso a
caso. Eso es lo que hay aquí, en unas pocas páginas de matemática que
puedes leer entera.

## Instalación

DualLoop es una librería, así que va dentro de un entorno virtual. Las dos
recetas de abajo crean `.venv` **en el directorio actual**, así que crea uno
para tu proyecto antes:

```bash
mkdir dualloop-demo && cd dualloop-demo
```

Con las herramientas estándar:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "dualloop @ git+https://github.com/JorgeTorresF/dualloop.git@v0.1.0"
```

O con [uv](https://github.com/astral-sh/uv):

```bash
uv venv && uv pip install "dualloop @ git+https://github.com/JorgeTorresF/dualloop.git@v0.1.0"
```

Dos cosas que conviene saber antes de ejecutarlas:

- **No las ejecutes desde tu carpeta personal.** Ahí apuntan a `~/.venv`, y
  si ya existe un entorno en esa ruta `uv` se ofrece a reemplazarlo.
  Responder que sí lo destruye, con todos sus paquetes dentro.
- **Un `pip install` a secas contra el Python del sistema será rechazado**
  en macOS con Homebrew y en casi toda distribución Linux actual, que lo
  marcan como gestionado externamente (PEP 668) con un error
  `externally-managed-environment`. Ese rechazo está haciendo su trabajo.
  En un Python de Homebrew puede que ni siquiera haya un `pip` en tu PATH,
  solo `pip3` y `python3 -m pip`.

Quita el `@v0.1.0` para seguir `master` en vez de la versión publicada.

Para desarrollo, clonando el repo:

```bash
git clone https://github.com/JorgeTorresF/dualloop.git
cd dualloop
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
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

Ver [`docs/architecture.es.md`](docs/architecture.es.md) para el detalle
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

El umbral está acotado por `threshold_lo` y `threshold_hi` (0.8 y 0.97 por
defecto). Cuando tus motores rinden mejor que `target_error_rate`, el
umbral baja hasta pegarse al suelo y se queda ahí — así que a partir de ese
punto es `threshold_lo`, y no la tasa de error objetivo, quien decide qué
se acepta. Fíjalo a conciencia.

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
- Sin `per_engine_correct`, `report_outcome()` solo puede deducir la
  verdad de los motores distintos del elegido cuando se sigue por lógica;
  donde no se sigue (un motor que discrepa en un `choice` de más de dos
  opciones), ese voto se deja sin actualizar en vez de adivinarlo. Pasa la
  verdad por motor cuando la tengas —por ejemplo, de un set de evaluación
  offline— para una recalibración más fina.
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
