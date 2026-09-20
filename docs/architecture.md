# Arquitectura de DualLoop

Este documento explica las decisiones de diseño y las conecta explícitamente
con los dos problemas de fondo identificados en la investigación previa
("Paradigmas faltantes en arquitectura IA"), citando fuente concreta para
cada afirmación.

## Los dos problemas de fondo

Dado un LLM de razonamiento (System 2) y un clasificador tipado rápido tipo
Jev (System 1), la investigación encontró que la pieza que falta no es un
tercer tipo de modelo, sino:

1. **Un árbitro aprendido y auditable entre motores heterogéneos.** La
   literatura de routing/cascading (RouteLLM, [arXiv:2410.10347](https://arxiv.org/abs/2410.10347);
   Meta-Reasoner, [arXiv:2502.19918](https://arxiv.org/abs/2502.19918); AAMC,
   [ScienceDirect S0925231226005898](https://www.sciencedirect.com/science/article/pii/S0925231226005898);
   CP-Router, arXiv:2505.19970) arbitra siempre entre **variantes de LLM**
   (fuerte/débil, LLM/LRM). Ningún trabajo revisado formaliza el arbitraje
   entre un LLM, un clasificador tipado no-generativo y reglas deterministas
   bajo un contrato común y auditable. Los propios autores de AAMC reconocen
   que sus resultados provienen de "experimentos de simulación de alta
   fidelidad, no de despliegues de producción reales".
2. **Cierre del bucle decisión → resultado → recalibración sin
   reentrenamiento manual.** Mem0 ("State of AI Agent Memory 2026",
   [mem0.ai/blog/state-of-ai-agent-memory-2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026))
   documenta que la memoria *procedimental* de agentes "sigue en etapa
   temprana". Los frameworks de "decision provenance" (PROV-AGENT; "A
   Framework for Assessing AI Agent Decisions and Outcomes in AutoML
   Pipelines", [arXiv:2602.22442](https://arxiv.org/html/2602.22442v1)) se
   detienen deliberadamente en auditoría para humanos: el propio paper
   aclara que su "Evaluation Agent" "no implementa auto-ajuste automatizado"
   y que sus salidas "apoyan la depuración humana en el bucle en lugar de la
   corrección autónoma de bucle cerrado". JitRL
   ([arXiv:2601.18510](https://arxiv.org/abs/2601.18510)) es el prototipo
   más cercano a una solución sin gradientes, pero requiere acceso a los
   logits del modelo (no funciona con APIs cerradas de caja negra) y se
   presenta como contribución de investigación, no como librería usable.

DualLoop ataca ambos problemas con matemática simple, interpretable y
sin gradientes — deliberadamente más humilde que un enfoque neuronal, a
cambio de funcionar con cualquier motor (incluidas APIs cerradas) y de ser
auditable por cualquiera que lea el código.

## Componentes

```
Question (contrato compartido: choice / score / noul, igual que simple-jev)
    │
    ▼
Arbiter.decide()
    │  1. ReliabilityBandit.rank()   -> orden de consulta (Thompson sampling
    │                                    Beta-Bernoulli, sesgo inicial por coste)
    │  2. engine.decide()             -> EngineOutput crudo del motor consultado
    │  3. ConfidenceCalibrator        -> confianza calibrada (histogram binning
    │                                    Beta-Bernoulli online)
    │  4. AdaptiveThreshold.get()     -> ¿aceptar o escalar al siguiente motor?
    ▼
Decision (registro auditable: votos, motor elegido, umbral usado, desacuerdo)
    │
    ▼  ... tiempo despues, se observa el resultado real ...
    ▼
DualLoop.report_outcome()
    │  actualiza en O(1), sin gradientes:
    │    - ConfidenceCalibrator.update()   (por motor y task_type)
    │    - ReliabilityBandit.update()      (fiabilidad del motor elegido)
    │    - AdaptiveThreshold.update_on_accepted_outcome()  (Robbins-Monro)
    ▼
Estado recalibrado, listo para la siguiente decision
```

### 1. Contrato compartido (`Question`)

Se reutiliza el esquema `choice`/`score`/`noul` de simple-jev para las tres
familias de motores (LLM, Jev, reglas). Esto es en sí mismo parte de la
pieza que falta: la investigación no encontró ningún framework de
orquestación mainstream que defina un contrato de decisión compartido entre
un LLM y un clasificador no-generativo — cada uno resuelve el routing solo
entre variantes de LLM (ver
[estado_actual_llm_clasificador.md](../../research_notes/Paradigmas%20faltantes%20en%20arquitectura%20IA/estado_actual_llm_clasificador.md)
en el informe de investigación previo).

### 2. Calibración por bins Beta-Bernoulli (`ConfidenceCalibrator`)

La confianza cruda de un clasificador tipado, la autoreportada de un LLM y
la certeza fija de una regla no son comparables sin calibrar. Se usa
histogram binning con actualización bayesiana cerrada:

- Cada bin de confianza `[i/n, (i+1)/n)` mantiene un posterior
  `Beta(alpha, beta)` sobre "¿acertó cuando reportó una confianza en este
  rango?".
- `calibrate(raw)` devuelve una mezcla entre la confianza cruda y la media
  del posterior del bin, con peso creciente hacia la evidencia real a
  medida que se acumulan observaciones (mezcla suave para evitar sobreajuste
  en frío).
- `update(raw, correct)` es la única operación necesaria para
  "recalibrarse solo": una actualización conjugada en O(1), sin gradientes
  ni reentrenamiento — lo suficientemente simple como para auditar leyendo
  el código fuente completo (< 80 líneas).

Esto es deliberadamente más simple que JitRL (que reponderar logits
requiere acceso al modelo) — el trade-off es menos elegante matemáticamente,
pero funciona con cualquier motor, incluidas APIs de caja negra.

### 3. Bandit de fiabilidad (`ReliabilityBandit`)

Determina el **orden** de la cascada (qué motor probar primero), no la
respuesta final. Thompson sampling Beta-Bernoulli por `(task_type, motor)`,
con un sesgo inicial optimista a favor de motores baratos (declarado vía
`relative_cost`) para que la exploración en frío no empiece al azar. El
sesgo se diluye en cuanto llegan outcomes reales — no es una regla fija de
"siempre empezar por el barato", es una prioridad inicial que el propio
bandit puede revertir si el motor barato resulta poco fiable para un
`task_type` concreto.

### 4. Umbral adaptativo (`AdaptiveThreshold`)

Aproximación estocástica de Robbins-Monro: el umbral de aceptación por
`task_type` se desplaza hacia el punto donde la tasa de error de las
respuestas aceptadas iguala una tasa objetivo configurable
(`target_error_rate`, 5% por defecto). Si el error observado supera el
objetivo, el umbral sube (más exigente, escala más); si es menor, baja
(menos escalamiento innecesario). Esto sustituye a un umbral fijo puesto a
mano, que es lo que hacen hoy los routers comerciales revisados
(RouteLLM, OpenRouter) — su umbral de "cuándo escalar" es un
hiperparámetro estático, no algo que se ajuste solo con la experiencia.

### 5. Auditoría (`Decision`, `explain()`)

Cada decisión registra: qué motores se consultaron y en qué orden, sus
salidas crudas y calibradas, el umbral vigente en ese momento, y si hubo
desacuerdo entre motores. `explain()` reconstruye todo esto más la
fiabilidad aprendida en el momento de la decisión. Esto responde
directamente al vacío de "decision provenance" documentado en
[arXiv:2602.22442](https://arxiv.org/html/2602.22442v1) y en el análisis de
TianPan.co sobre "Decision Provenance in Agentic Systems"
([tianpan.co](https://tianpan.co/blog/2026-04-19-decision-provenance-agentic-systems)),
que concluye que "la respuesta para la mayoría de sistemas agénticos en
producción hoy es no" cuando se pregunta si pueden reconstruir cadenas de
decisión para rendir cuentas.

## Por qué NO se implementó...

- **Reponderación de logits (estilo JitRL):** requiere acceso a los pesos o
  logits del modelo. Rompe con LLMs vía API cerrada (OpenAI, Anthropic) y
  con Jev vía HTTP. Se prefirió un mecanismo que funcione con cualquier
  motor tratado como caja negra, al coste de ser menos preciso que
  reponderar logits directamente.
- **World models / razonamiento causal:** según la misma investigación
  previa, sigue siendo investigación básica sin producto maduro (JEPA de
  LeCun, AMI Labs, financiado con $1.03B pero "sin producto ni ingresos" a
  fecha de esa investigación) — fuera de alcance de una librería que se
  quiere usable hoy.
- **Verificación formal (SMT solvers, Lean) como motor adicional:** dejado
  como extensión del `Engine` protocol para quien lo necesite (ver
  Roadmap en el README) — la investigación encontró aplicaciones
  incipientes fuera de matemáticas/código (p.ej. razonamiento legal,
  [arXiv:2511.21033](https://arxiv.org/abs/2511.21033)) pero ninguna en
  decisiones clínicas o de negocio general, así que no se incluyó una
  implementación concreta para no inventar un caso de uso no validado.
- **Bucketing de contexto por embeddings:** el MVP bucketiza solo por
  `task_type` explícito (declarado por quien llama) en vez de por
  similitud semántica entre casos, para mantener cero dependencias de
  modelos de embeddings. Es el principal punto de extensión documentado en
  el Roadmap.

## Validación empírica

`examples/demo_synthetic.py` simula 800 decisiones con tres motores de
fiabilidad y coste distintos (uno deliberadamente sobreconfiado en casos
difíciles) y mide, por ventanas de 50 decisiones: precisión de la respuesta
aceptada, número medio de motores consultados por decisión (proxy de
coste), y error de calibración (|confianza calibrada − acierto real|). En
una corrida de referencia con semilla fija, el error de calibración bajó de
~0.15 a ~0.06 a lo largo de la corrida sin ninguna intervención manual,
mientras la precisión se mantuvo estable y el número de motores consultados
por decisión se mantuvo bajo (~1.1-1.2 de media, es decir, la cascada
resuelve la mayoría de los casos con un solo motor).

Esto es una prueba de que el mecanismo funciona como está diseñado sobre
datos sintéticos — no una validación en producción ni sobre un dominio
real. Ver la sección "Limitaciones honestas" del README.
