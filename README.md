# DualLoop

[![CI](https://github.com/JorgeTorresF/dualloop/actions/workflows/ci.yml/badge.svg)](https://github.com/JorgeTorresF/dualloop/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Learned, auditable arbitration between heterogeneous decision engines (a
reasoning LLM, a typed classifier such as
[Jev/simple-jev](https://github.com/featherless-ai/simple-jev),
deterministic rules...), closing the loop **decision → outcome →
recalibration, with no manual retraining**.

*Lee este README [en español](README.es.md).*

## Why this exists

Pairing a "System 2" LLM (slow, expensive, general reasoning) with a fast
typed "System 1" classifier is now common —
[TypeSafe AI/Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
launched in September 2026 positioning itself exactly that way. What does
**not exist yet, neither in mainstream orchestration frameworks (LangGraph,
CrewAI, AutoGen, Semantic Kernel) nor in commercial LLM routers**
(RouteLLM, OpenRouter, Martian, Not Diamond — all of which route only
between LLM variants, never between engines of genuinely different kinds)
are two specific pieces:

1. **A learned, auditable arbiter across heterogeneous engines.** No
   reviewed work arbitrates between an LLM, a typed classifier and
   deterministic rules under one common contract. Existing
   meta-controllers route between **LLM variants** — AAMC
   ([ScienceDirect S0925231226005898](https://www.sciencedirect.com/science/article/pii/S0925231226005898))
   orchestrates SLM/LLM with reinforcement learning — or not even that:
   Meta-Reasoner ([arXiv:2502.19918](https://arxiv.org/abs/2502.19918))
   operates *within* a single model, using contextual bandits to decide
   when to backtrack, switch approach or restart its reasoning. It is
   DualLoop's closest relative in using bandits for meta-decisions, but
   its action space is reasoning strategies, not engines.
2. **Closing the loop without manual retraining.** Research on agent
   memory (Mem0, ["State of AI Agent Memory 2026"](https://mem0.ai/blog/state-of-ai-agent-memory-2026))
   confirms that *procedural* memory — learning from outcomes — "remains
   in an early stage"; "decision provenance" frameworks
   ([arXiv:2602.22442](https://arxiv.org/html/2602.22442v1)) deliberately
   stop at auditability for humans and never reach self-adjustment; and
   the closest thing to a gradient-free solution (JitRL,
   [arXiv:2601.18510](https://arxiv.org/abs/2601.18510)) is still a
   research prototype rather than a usable library.

DualLoop does not solve this with a new model, but with simple, auditable
maths (Bayesian histogram calibration + Beta-Bernoulli bandits +
constant-step stochastic approximation) that works with **any engine,
including closed black-box APIs** — it needs no access to logits or model
weights, unlike JitRL. The full reasoning behind each design decision, with
sources, is in [`docs/architecture.md`](docs/architecture.md).

## Is this for you?

DualLoop is not an agent framework and not an LLM router. It is a small
piece for one specific problem: **you have several ways to make the same
decision, with different cost and reliability, and you want to use the
cheap one when it suffices and the expensive one when it doesn't — without
hand-tuning where that line sits.**

It fits if your case meets all four:

1. **The decision repeats.** Tens or hundreds of cases of the same kind,
   not a one-off deliberation.
2. **The answer is typed.** One option among several, a score on a rubric,
   a yes/no judgment. Not free prose.
3. **The truth eventually arrives.** Someone reviews it, the case
   resolves, the customer replies. In hours or days, not months.
4. **Your engines are unequal.** A small classifier and an expensive API;
   or rules and an LLM. With a single engine there is nothing to arbitrate.

Examples that meet all four:

- **Content review gates.** Posts, reports or replies that pass a filter
  before going out; whoever reviews confirms or corrects, and that
  correction is the ground truth.
- **Ticket or incident triage.** Category and priority; the truth is where
  the ticket actually ended up.
- **Moderation and abuse detection.** Cheap rules for the obvious,
  a classifier for volume, an LLM for the ambiguous.
- **Quality control on data extraction.** Is this extracted field right?
  Human spot-checks feed back into the system.
- **CI gates.** Does this change need a human reviewer? The truth is
  whether the reviewer found anything.
- **Lead qualification or spam filtering**, where the outcome becomes
  known shortly afterwards.

And the cases that **don't** fit, to save you the disappointment:
decisions you make five times a year; truths that take months to arrive;
free-prose outputs; or a single engine. In all of those the calibrator
never leaves cold start and you will not notice any difference from a
hand-written fixed threshold.

**What it gives you over rolling your own.** You could write the
`if confidence > 0.8` yourself. What is not trivial is the rest: making
confidence from different engines comparable at all, having that `0.8`
move on its own according to the errors you actually make, having the
consultation order learn which engine is reliable for each task type, and
keeping every case auditable afterwards. That is what lives here, in a few
pages of maths you can read end to end.

## Installation

DualLoop is a library, so it belongs in a virtual environment. Both recipes
below create `.venv` **in the current directory**, so make one for your
project first:

```bash
mkdir dualloop-demo && cd dualloop-demo
```

With the standard tooling:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "dualloop @ git+https://github.com/JorgeTorresF/dualloop.git@v0.1.0"
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv venv && uv pip install "dualloop @ git+https://github.com/JorgeTorresF/dualloop.git@v0.1.0"
```

Two things worth knowing before you run them:

- **Do not run these from your home directory.** There they target
  `~/.venv`, and if an environment already exists at that path `uv` offers
  to replace it. Answering yes destroys it, every installed package with
  it.
- **A bare `pip install` into the system Python will be refused** on
  Homebrew macOS and on most current Linux distributions, which mark it
  externally managed (PEP 668) with an `externally-managed-environment`
  error. That refusal is doing its job. On a Homebrew Python there may also
  be no `pip` on your PATH at all, only `pip3` and `python3 -m pip`.

Drop the `@v0.1.0` to track `master` instead of the released version.

For development, cloning the repo:

```bash
git clone https://github.com/JorgeTorresF/dualloop.git
cd dualloop
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

No required dependencies beyond `httpx`. The `demo` extra adds
`matplotlib` (optional, only to plot the synthetic example).

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
    # A deterministic rule beats any model when the case is obvious: an
    # explicit dosage means a prescription, and there is nothing to weigh.
    RuleEngine(lambda task_type, q: ("prescribes", 0.99) if " mg" in q.state else None),
]

loop = DualLoop(engines=engines)

# A style gate on a report a patient will read: the report recommends and
# refers to a doctor, it never prescribes.
question = Question(
    type="choice",
    instructions="Does this paragraph prescribe a treatment, or merely recommend?",
    state="You may want to discuss with your doctor whether your current plan needs review.",
    criteria={"recommends": None, "prescribes": None},
)

decision = loop.decide(task_type="prescription_gate", question=question)
print(decision.chosen_engine, decision.chosen_value, decision.chosen_confidence)

# ... once the human reviewer confirms or corrects the verdict ...
loop.report_outcome(decision.id, correct=True)
```

**Why this example and not another.** DualLoop learns from outcomes, so it
only pays off where outcomes arrive soon and in quantity: tens or hundreds
of cases per task type, with the truth known in hours or days. A review
gate over documents that are already reviewed daily fits. A decision made
five times a year whose truth takes months — assessing a grant
application, say — does not: the calibrator never leaves cold start and you
will not notice any difference from a fixed rule.

Each `report_outcome()` recalibrates engine confidence online for that task
type, adjusts how reliable each engine looks, and moves the acceptance
threshold — without touching any model weight or retraining anything.

For Anthropic/Claude instead of an OpenAI-compatible endpoint:

```python
from dualloop.engines import AnthropicLLMEngine

llm = AnthropicLLMEngine(api_key="sk-ant-...", model="claude-sonnet-5")
```

To persist state across process restarts:

```python
from dualloop import SQLiteStore

loop = DualLoop(engines=engines, store=SQLiteStore("dualloop.db"))
```

## How it works (in brief)

1. **`decide()`** ranks the available engines by learned reliability
   (Thompson sampling bandit), consults the most promising first,
   calibrates its raw confidence and, if that clears the task type's
   adaptive threshold, accepts the answer without escalating further.
   Otherwise it tries the next engine. Everything is recorded in an
   auditable `Decision`.
2. **`report_outcome()`** updates three online statistics in a single
   call: the confidence calibration of the engine used, its aggregate
   reliability for that task type, and the acceptance threshold — all with
   closed-form Bayesian updates in O(1), no gradients.
3. **`explain()`** reconstructs why a decision was made: which engines
   were consulted, their raw and calibrated votes, the threshold in force,
   and the reliability learned at that moment.

See [`docs/architecture.md`](docs/architecture.md) for the mathematical
detail and the justification of each design choice.

## Abstention: when the system does not know

By default, if no engine clears the threshold, the loop still accepts the
highest calibrated vote: the threshold is recorded on the decision but does
not act as a floor. For a gate where "I don't know, let a person look at
it" is a legitimate answer, that will not do:

```python
loop = DualLoop(engines=engines, abstain_below_threshold=True)

decision = loop.decide(task_type="prescription_gate", question=question)
if decision.abstained:
    send_to_human_review(decision)   # chosen_value is still available
else:
    apply(decision.chosen_value)
```

An abstained decision **does not move the acceptance threshold** when you
report its outcome: the threshold targets the error rate among *accepted*
answers, and an abstention was not accepted — raising the bar because of a
case the system had already flagged as doubtful would break that
semantics. The calibrators and the bandit do learn from it.

The threshold is bounded by `threshold_lo` and `threshold_hi` (0.8 and
0.97 by default). When your engines perform better than
`target_error_rate`, the threshold drifts down until it rests on the floor
and stays there — so past that point it is `threshold_lo`, not the target
error rate, that decides what gets accepted. Set it deliberately.

## Synthetic demo

```bash
python examples/demo_synthetic.py
```

Simulates 800 decisions across three engines of differing reliability and
cost (one of them deliberately overconfident on hard cases) and shows how
accuracy, cost (engines consulted per decision) and calibration error
evolve with experience, with no manual intervention in between.

## Honest limitations

- The demo is **synthetic**: it shows the recalibration mechanism behaves
  as designed, not that it solves any real domain. It has not been
  validated in production.
- `JevEngine`'s parsing of `noul` responses is inferred from simple-jev's
  public specification, not confirmed against a real response — check it
  against your own server's `/docs` before relying on it in production.
- Without `per_engine_correct`, `report_outcome()` can only infer the
  truth for engines other than the chosen one in cases where it follows
  logically; where it does not (a dissenting engine on a `choice` with
  more than two options), that vote is left un-updated rather than guessed
  at. Pass the per-engine truth when you have it — for instance from an
  offline evaluation set — for a sharper recalibration.
- The licence of the `simple-jev` server itself was not clearly stated in
  its repository at the time of this research (there is an open issue
  about it) — this client only speaks its public HTTP protocol and
  redistributes none of its code, but verify the licence of your own
  deployment separately.
- Context bucketing is by explicit `task_type`, not by semantic similarity
  between cases (see Roadmap).

## Roadmap / possible extensions

- Context bucketing by similarity (embeddings) rather than `task_type`
  alone, to generalise calibration across similar cases.
- Additional store backends (Postgres, Redis) implementing the same
  `Store` protocol.
- Formal verification as an additional engine for domains with explicit
  logical structure (code, regulatory compliance).
- Further outcome-inference heuristics, documented with their limitations
  as explicitly as `HumanOverrideHeuristic`.

## Licence

MIT — see [`LICENSE`](LICENSE).
