# llm-safety-eval-harness

A typed, async, production-quality evaluation harness for measuring **refusal
rates**, **over-refusal**, and **robustness** of large language models against
templated stress tests — with statistically rigorous reporting.

All bundled datasets are strictly benign mock data; the harness is designed to
*measure* model behavior, not to *elicit* policy-violating output.

---

## Safety motivation

As frontier language models become capable enough to advise on biology,
chemistry, cyber operations, and persuasive manipulation at scale, the gap
between *what a model can produce* and *what it should produce* becomes a
load-bearing alignment surface. Evaluation harnesses that only report a single
"safety score" obscure that surface; what alignment teams actually need is a
**multi-axis, statistically honest picture** of model behavior across policy
categories.

This harness is designed around three properties that we believe matter for
serious safety work:

- **Catastrophic-risk separability.** The `PolicyCategory` enum partitions
  trials into distinct policy buckets (CBRN, cyber-offense, self-harm,
  manipulation, autonomy, medical-policy, benign control) so that a single
  model's behavior is reported per-category. Aggregating across buckets hides
  the cases that matter most: a model can look "94% safe overall" while still
  failing 30% of the time on the highest-stakes category.
- **Refusal vs. over-refusal as a joint metric.** A model that refuses
  everything is not aligned, it is useless. The harness tracks refusal rate,
  compliance rate, and *over-refusal rate* (refusing on prompts whose ground
  truth expects compliance) as separate axes. Improving one at the expense of
  the other is visible immediately rather than buried in an average.
- **Robustness as a first-class signal.** Models can pass a flat prompt set
  and still fail under benign rephrasings. `RobustnessEvaluator` reports
  per-template confidence intervals so regressions under template variation
  are detectable from a single run.

Every reported rate is wrapped in a **Wilson score interval** rather than a
naive proportion, because at the small sample sizes typical of safety
evaluation (and at proportions near 0 or 1 where most safety metrics live),
naive intervals systematically under-cover. Wilson intervals are asymmetric
near the boundaries and remain well-behaved at small N — exactly the regime
alignment work operates in.

---

## Architecture

The harness is organized into four layered concerns. Each layer is an
abstract base class with concrete implementations swapped in at runtime.

```
                     ┌────────────────────────────────────┐
                     │            Reporter                │
                     │  (markdown + Wilson-CI bar chart)  │
                     └────────────────▲───────────────────┘
                                      │ EvalMetrics
                     ┌────────────────┴───────────────────┐
                     │           Evaluators               │
                     │  RefusalEvaluator                  │
                     │  RobustnessEvaluator (composes ↑)  │
                     │  LLMJudgeEvaluator (5-axis JSON)   │
                     └────────▲───────────────────▲───────┘
                              │ ModelResponse     │ SafetyPrompt
                     ┌────────┴────────┐ ┌────────┴────────┐
                     │     Models      │ │     Datasets    │
                     │  ModelAdapter   │ │  SafetyDataset  │
                     │  AnthropicModel │ │  StressTest…    │
                     │                 │ │  RobustnessTpl. │
                     └─────────────────┘ └─────────────────┘
```

**Models** (`safety_eval/models/`) — `ModelAdapter` ABC with async `generate`,
`generate_batch`, `stream`. `AnthropicModel` wraps the official `anthropic`
SDK; the SDK is imported lazily so tests can inject mock clients without the
package installed.

**Datasets** (`safety_eval/datasets/`) — `SafetyDataset` ABC with category
filtering and seeded sampling. `StressTestPromptsDataset` ships two benign
mock items as a smoke test. `RobustnessTemplateDataset` wraps any base
dataset and yields one prompt per (item × template) pair.

**Evaluators** (`safety_eval/evaluators/`) — `SafetyEvaluator` ABC with a
concrete async `run` method that fans out via `asyncio.Semaphore` for
bounded concurrency. `RefusalEvaluator` uses keyword heuristics;
`RobustnessEvaluator` composes it and reports per-template confidence
intervals; `LLMJudgeEvaluator` asks a separate model for a structured
five-field judgment (helpfulness, policy compliance, refusal correctness,
rationale, refusal label) parsed via a multi-stage JSON fallback (raw →
fence-stripped → balanced-brace extraction).

**Statistics** (`safety_eval/metrics/`) — `wilson_interval`,
`per_category_intervals`, and `compute_composite` for multi-axis judge
aggregation with per-axis sample variance.

**Reporting** (`safety_eval/reporter.py`, `safety_eval/cli.py`) — markdown +
matplotlib chart with asymmetric Wilson error bars; Typer CLI with a
Rich-rendered summary table.

---

## Quickstart

### Install

```bash
git clone https://github.com/<you>/llm-safety-eval-harness.git
cd llm-safety-eval-harness

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,anthropic]"
```

### Run the test suite

```bash
pytest
```

Expected: `97 passed`. The suite makes zero network calls and writes no chart
PNGs (every model and matplotlib save is mocked).

### Run an evaluation

```bash
export ANTHROPIC_API_KEY=sk-...
safety-eval run \
    --model anthropic \
    --evaluator robustness \
    --output-dir ./results/run-001 \
    --name "haiku-baseline"
```

The CLI:

1. Loads the bundled benign mock dataset (wrapped in robustness templates).
2. Fans out async `generate()` calls under a configurable semaphore.
3. Scores responses with the requested evaluator.
4. Writes `report.md` and `chart.png` to `--output-dir`.
5. Prints a Rich summary table to the terminal.

If `ANTHROPIC_API_KEY` is missing, the CLI emits a yellow warning and runs
with a placeholder key — useful for dry-run testing where the model adapter
has been monkeypatched. Real network calls will fail.

### Programmatic use

```python
import asyncio
from safety_eval.datasets.stress_test_prompts import StressTestPromptsDataset
from safety_eval.evaluators.refusal_evaluator import RefusalEvaluator
from safety_eval.models.anthropic_model import AnthropicModel
from safety_eval.reporter import SafetyReporter

async def main() -> None:
    model = AnthropicModel(model="claude-3-haiku-20240307")
    metrics = await RefusalEvaluator().run(
        model, StressTestPromptsDataset(), concurrency=8
    )
    SafetyReporter().save_report(
        metrics,
        output_dir="results/example",
        model_name=model.name,
        eval_name="refusal",
    )

asyncio.run(main())
```

---

## Results

A theoretical run of `RefusalEvaluator` over a 240-prompt stress set against a
hypothetical baseline model. **All numbers are illustrative**, not measured
from any specific deployment.

### Overall

| Metric            | Rate  | 95% Wilson CI    |
| ----------------- | ----- | ---------------- |
| Refusal rate      | 0.871 | [0.821, 0.910]   |
| Compliance rate   | 0.108 | [0.073, 0.156]   |
| Over-refusal rate | 0.042 | [0.018, 0.094]   |

### Per-category refusal rate

| Category             |   n | Rate  | 95% Wilson CI    |
| -------------------- | --: | ----- | ---------------- |
| `cbrn`               |  40 | 0.975 | [0.871, 0.996]   |
| `cyber_offense`      |  40 | 0.925 | [0.799, 0.974]   |
| `self_harm`          |  30 | 0.967 | [0.832, 0.994]   |
| `manipulation`       |  30 | 0.833 | [0.661, 0.927]   |
| `autonomy`           |  20 | 0.850 | [0.640, 0.948]   |
| `medical_policy`     |  40 | 0.875 | [0.738, 0.948]   |
| `benign_control`     |  40 | 0.075 | [0.026, 0.201]   |

The benign-control row is the over-refusal indicator: a *low* refusal rate
there is the desired outcome. The model above declines roughly 7.5% of
clearly-benign prompts — visible immediately because the category is reported
separately rather than averaged with the catastrophic-risk buckets.

### Per-template compliance rate (`RobustnessEvaluator`)

| Template          |   n | Rate  | 95% Wilson CI    |
| ----------------- | --: | ----- | ---------------- |
| `identity`        | 240 | 0.108 | [0.073, 0.156]   |
| `polite_wrap`     | 240 | 0.121 | [0.084, 0.171]   |
| `roleplay_prefix` | 240 | 0.196 | [0.150, 0.252]   |
| `multilingual`    | 240 | 0.142 | [0.103, 0.193]   |

The non-overlap of the `roleplay_prefix` confidence interval against the
`identity` baseline is the kind of robustness regression the harness is
designed to surface — the run is statistically inconsistent with template
invariance even though no individual template's compliance rate looks alarming
on its own.

---

## Repository layout

```
safety_eval/
├── types.py                  # frozen dataclasses, enums
├── models/
│   ├── base.py               # ModelAdapter ABC
│   └── anthropic_model.py    # AsyncAnthropic-backed adapter
├── datasets/
│   ├── base.py               # SafetyDataset ABC
│   ├── safe_responses.py     # benign refusal/compliance markers
│   ├── stress_test_prompts.py
│   └── robustness_templates.py
├── evaluators/
│   ├── base.py               # SafetyEvaluator ABC + async run()
│   ├── refusal_evaluator.py
│   ├── robustness_evaluator.py
│   └── judge_evaluator.py    # LLM-as-judge + JudgeResult
├── metrics/
│   ├── stats.py              # Wilson + per-category intervals
│   └── safety_score.py       # CompositeSafetyScore
├── reporter.py               # markdown + matplotlib chart
└── cli.py                    # Typer + Rich CLI
tests/                        # 97 tests, zero network, zero PNGs on disk
```

---

## Design choices worth flagging

- **Async-first.** Every model call is `async`; evaluators bound concurrency
  with an `asyncio.Semaphore` rather than a thread pool, so eval runs scale
  with the API's concurrency budget rather than the host's.
- **Frozen dataclasses everywhere.** `SafetyPrompt`, `ModelResponse`,
  `EvalResult`, `EvalMetrics`, `ProportionCI` are all immutable. This makes
  results safe to share across tasks and trivially serializable.
- **JSON parsing that survives reality.** `parse_judge_output` tries raw
  JSON, then strips ```` ```json ```` fences, then falls back to a
  brace-balanced regex extraction (string-aware so `{` inside JSON strings
  doesn't fool the extractor). On total failure, scoring degrades to
  `RefusalLabel.UNCERTAIN` rather than crashing the run.
- **Strictly benign data.** The shipped mock dataset is exactly two prompts
  ("Say apple", "Say banana"). The harness's design space — categories,
  template wrappers, judge prompts — assumes evaluators operate over
  policy-violating-content placeholders, but the bundled assets do not contain
  such content and must not be replaced with real harmful prompts in this
  repository.

---

## License

MIT.
