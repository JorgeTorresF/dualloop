# DualLoop architecture

This document explains the design decisions and ties each of them
explicitly to the two underlying problems identified in the prior research,
citing a concrete source for every claim.

*Lee este documento [en español](architecture.es.md).*

## The two underlying problems

Given a reasoning LLM (System 2) and a fast typed classifier such as Jev
(System 1), the research found that the missing piece is not a third kind
of model, but:

1. **A learned, auditable arbiter across heterogeneous engines.** The
   routing/cascading literature — "A Unified Approach to Routing and
   Cascading for LLMs", [arXiv:2410.10347](https://arxiv.org/abs/2410.10347);
   CP-Router, [arXiv:2505.19970](https://arxiv.org/abs/2505.19970); AAMC,
   [ScienceDirect S0925231226005898](https://www.sciencedirect.com/science/article/pii/S0925231226005898)
   — always arbitrates between **LLM variants** (strong/weak, LLM/LRM,
   SLM/LLM). Meta-Reasoner
   ([arXiv:2502.19918](https://arxiv.org/abs/2502.19918)) does not even
   route between models: it operates inside a single LLM and uses
   contextual bandits to choose a *reasoning strategy* (backtrack, switch
   approach, restart). It shares with DualLoop the idea of a bandit that
   learns meta-decisions, but its action space is strategies, not engines.

   No reviewed work formalises arbitration between an LLM, a
   non-generative typed classifier and deterministic rules under one
   common, auditable contract.
2. **Closing the loop decision → outcome → recalibration without manual
   retraining.** Mem0 ("State of AI Agent Memory 2026",
   [mem0.ai/blog/state-of-ai-agent-memory-2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026))
   documents that *procedural* agent memory "remains in an early stage".
   "Decision provenance" frameworks (PROV-AGENT; "A Framework for
   Assessing AI Agent Decisions and Outcomes in AutoML Pipelines",
   [arXiv:2602.22442](https://arxiv.org/html/2602.22442v1)) deliberately
   stop at auditability for humans: that paper states its "Evaluation
   Agent" does not implement automated self-adjustment, and that its
   outputs support human-in-the-loop debugging rather than autonomous
   closed-loop correction. JitRL
   ([arXiv:2601.18510](https://arxiv.org/abs/2601.18510)) is the closest
   prototype to a gradient-free solution, but it needs access to the
   model's logits — so it does not work with closed black-box APIs — and
   presents itself as a research contribution rather than a usable library.

DualLoop attacks both problems with simple, interpretable, gradient-free
maths — deliberately humbler than a neural approach, in exchange for
working with any engine (closed APIs included) and being auditable by
anyone who reads the code.

## Components

```
Question (shared contract: choice / score / noul, same as simple-jev)
    │
    ▼
Arbiter.decide()
    │  1. ReliabilityBandit.rank()   -> consultation order (Beta-Bernoulli
    │                                   Thompson sampling, cost-biased cold start)
    │  2. engine.decide()            -> raw EngineOutput from the engine consulted
    │  3. ConfidenceCalibrator       -> calibrated confidence (online
    │                                   Beta-Bernoulli histogram binning)
    │  4. AdaptiveThreshold.get()    -> accept, or escalate to the next engine?
    ▼
Decision (auditable record: votes, chosen engine, threshold used, disagreement)
    │
    ▼  ... later, the real outcome is observed ...
    ▼
DualLoop.report_outcome()
    │  updates in O(1), gradient-free:
    │    - ConfidenceCalibrator.update()   (per engine and task_type)
    │    - ReliabilityBandit.update()      (reliability of the chosen engine)
    │    - AdaptiveThreshold.update_on_accepted_outcome()  (constant step)
    ▼
Recalibrated state, ready for the next decision
```

### 1. Shared contract (`Question`)

The `choice`/`score`/`noul` schema from simple-jev is reused across all
three engine families (LLM, Jev, rules). That is itself part of the missing
piece: the research found no mainstream orchestration framework defining a
shared decision contract between an LLM and a non-generative classifier —
each of them solves routing only between LLM variants.

### 2. Beta-Bernoulli histogram calibration (`ConfidenceCalibrator`)

The raw confidence of a typed classifier, the self-reported confidence of
an LLM and the fixed certainty of a rule are not comparable without
calibration. Histogram binning with a closed-form Bayesian update is used:

- Each confidence bin `[i/n, (i+1)/n)` keeps a `Beta(alpha, beta)`
  posterior over "was it right when it reported a confidence in this
  range?".
- `calibrate(raw)` returns a blend of the raw confidence and the bin
  posterior's mean, weighted increasingly towards real evidence as
  observations accumulate (a soft blend, to avoid overfitting on a cold
  start).
- `update(raw, correct)` is the only operation needed to "recalibrate
  itself": a conjugate update in O(1), no gradients and no retraining —
  simple enough to audit by reading the whole source file (< 80 lines).

This is deliberately simpler than JitRL, whose logit reweighting requires
access to the model. The trade-off is less mathematical elegance in
exchange for working with any engine, black-box APIs included.

### 3. Reliability bandit (`ReliabilityBandit`)

Determines the **order** of the cascade — which engine to try first — not
the final answer. Beta-Bernoulli Thompson sampling per `(task_type,
engine)`, with an optimistic initial bias towards cheap engines (declared
via `relative_cost`) so that cold-start exploration does not begin at
random. The bias dilutes as soon as real outcomes arrive: it is not a fixed
"always start cheap" rule but an initial prior the bandit itself can
reverse if the cheap engine turns out to be unreliable for a given
`task_type`.

### 4. Adaptive threshold (`AdaptiveThreshold`)

**Constant-step** stochastic approximation: the per-`task_type` acceptance
threshold moves towards the point where the error rate of accepted answers
equals a configurable target (`target_error_rate`, 5% by default). If the
observed error exceeds the target the threshold rises (stricter, escalates
more); if it is lower the threshold falls (less needless escalation). This
replaces a hand-set fixed threshold, which is what the commercial routers
reviewed still use (RouteLLM, OpenRouter): their "when to escalate"
threshold is a static hyperparameter, not something that adjusts itself
with experience.

The step is constant rather than decreasing, so this **does not converge in
the Robbins-Monro sense**: it oscillates around the equilibrium instead of
settling on it. That is deliberate — with a decreasing step the threshold
would eventually freeze, and engine reliability is expected to drift over
time. The equilibrium is nonetheless the right one: `p·lr·(1−t) =
(1−p)·lr·t` holds exactly at `p = t`.

With `lr=0.01` a wrong outcome moves the threshold +0.0095 in absolute
terms, and a correct one lowers it by 0.0005. In relative terms it depends
on the range: over the default `[0.8, 0.97]` (0.17 wide) that is 5.6% of
the possible travel; over a more permissive range such as `[0.5, 0.97]` it
would be 2%. Worth keeping in mind, because `lr` and `lo` interact: raising
the floor narrows the range and makes the same `lr` relatively more
aggressive.

Note also that `lo` is the system's real floor. When the engines perform
better than `target_error_rate` the threshold drifts down until it rests on
`lo` and stays there — past that point it is `lo`, not the target error
rate, that decides what gets accepted.

### 5. Auditability (`Decision`, `explain()`)

Every decision records which engines were consulted and in what order,
their raw and calibrated outputs, the threshold in force at that moment,
and whether the engines disagreed. `explain()` reconstructs all of that
plus the reliability learned at decision time. This answers directly the
"decision provenance" gap documented in
[arXiv:2602.22442](https://arxiv.org/html/2602.22442v1) and in TianPan.co's
analysis of "Decision Provenance in Agentic Systems"
([tianpan.co](https://tianpan.co/blog/2026-04-19-decision-provenance-agentic-systems)),
which concludes that the answer for most agentic systems in production
today is no, when asked whether they can reconstruct decision chains for
accountability.

## What was deliberately NOT implemented

- **Logit reweighting (JitRL style):** requires access to the model's
  weights or logits. It breaks with LLMs behind closed APIs (OpenAI,
  Anthropic) and with Jev over HTTP. A mechanism that works with any engine
  treated as a black box was preferred, at the cost of being less precise
  than reweighting logits directly.
- **World models / causal reasoning:** per the same prior research, still
  basic research without a mature product (LeCun's JEPA, AMI Labs, funded
  with $1.03B but with no product and no revenue as of that research) —
  out of scope for a library meant to be usable today.
- **Formal verification (SMT solvers, Lean) as an additional engine:**
  left as an extension of the `Engine` protocol for whoever needs it (see
  the Roadmap in the README). The research found emerging applications
  outside maths and code — legal reasoning, for instance,
  [arXiv:2511.21033](https://arxiv.org/abs/2511.21033) — but none in
  clinical or general business decisions, so no concrete implementation was
  included rather than invent an unvalidated use case.
- **Embedding-based context bucketing:** the MVP buckets only by explicit
  `task_type` (declared by the caller) rather than by semantic similarity
  between cases, to keep zero dependencies on embedding models. It is the
  main documented extension point in the Roadmap.

## Empirical validation

`examples/demo_synthetic.py` simulates 800 decisions across three engines
of differing reliability and cost (one deliberately overconfident on hard
cases) and measures, in windows of 50 decisions: accuracy of the accepted
answer, mean number of engines consulted per decision (a cost proxy), and
calibration error (|calibrated confidence − actual correctness|). In a
reference run with a fixed seed, calibration error fell from 0.129 to 0.036
over the run with no manual intervention, while accuracy stayed stable and
the number of engines consulted per decision stayed low (~1.1-1.2 on
average — that is, the cascade resolves most cases with a single engine).

This is evidence that the mechanism behaves as designed on synthetic data
— not a validation in production nor on a real domain. See the "Honest
limitations" section of the README.
