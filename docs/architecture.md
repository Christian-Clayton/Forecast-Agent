# forecast-agent Architecture

## Overview

`forecast-agent` is a tool-using forecasting agent with deterministic actuarial and statistical tools, an append-only forecast ledger, and a scoring harness. The LLM never does arithmetic; every number in a forecast must come from a tool call that executed code.

## Components

### LLM provider interface

- `provider.py` -- abstract interface with `complete(messages, model, max_tokens)`
- `openai_compat_provider.py` -- OpenAI-compatible HTTP provider

### Quant tools (MCP server)

`quant_server.py` exposes pure functions as MCP tools:

- `baseline_forecast` -- naive, seasonal_naive with empirical intervals
- `fit_ets` / `fit_arima` -- time-series models with intervals
- `walk_forward_score` -- cross-validated error metric
- `fit_frequency_severity` -- Poisson/NegBin + Gamma/lognormal
- `aggregate_loss_mc` -- Monte Carlo aggregate loss
- `buhlmann_credibility` -- Buhlmann credibility weighting
- `chain_ladder` / `bornhuetter_ferguson` -- claims reserving
- `evt_pot_fit` -- peaks-over-threshold extreme value theory
- `ensemble_aggregate` -- median, trimmed mean, logit mean, spread

### Data server

`data_server.py` provides as-of-gated access to synthetic telecom telemetry:

- `list_series` -- available series with metadata
- `get_series` -- time-series observations (date-gated)
- `get_event_table` -- event tables with severity
- `get_reference_class` -- base rate references

### Ledger server

`ledger_server.py` exposes the append-only forecast ledger as MCP tools:

- `write_question` / `write_forecast` -- insert only
- `read_forecast` / `list_open_questions` -- read access
- `record_resolution` / `score_forecast` -- lifecycle
- DB triggers enforce append-only on `forecasts`

### Search server (phase 7)

`search_server.py` provides web search for live questions (synthetic implementation).

### Agent pipeline

`agent/pipeline.py` orchestrates 8 stages, each a pydantic contract:

1. **Intake** -- restate as falsifiable
2. **Classify** -- thin / fat / mixed
3. **Base rate** -- reference class + base rate
4. **Decompose** -- sub-questions + series selection
5. **Model** -- run tools, always baseline first
6. **Ensemble** -- N members, spread rule to Toolkit B
7. **Critique** -- pre-mortem, disconfirming hypotheses
8. **Finalize** -- actuarial style (bottom line first)

Budget caps (steps, tokens, cost) apply per question. Skills load on demand via `skills/loader.py`.

## Data flow

```
Question
  -> Intake (falsifiable restatement)
  -> Classify (thin | fat | mixed)
  -> Base rate (reference class, base rate, source)
  -> Decompose (sub-questions, series, horizon)
  -> Model (baseline first, then complex)
  -> Ensemble (N members, spread check)
  -> Critique (pre-mortem, disconfirming hypotheses)
  -> Finalize (actuarial style, citations)
  -> Ledger (append-only write)
  -> Score (Brier, log, pinball)
```

## Evaluation harness

Run all evals: `src/forecast_agent/evals/run_all.py`

| Eval | What it measures | Pass |
|------|-----------------|------|
| Walk-forward backtest | Error vs naive and seasonal-naive | Report results |
| Quant correctness | Tool outputs vs known-answer fixtures | Match to tolerance |
| Classifier eval | Thin vs fat labelling | Report accuracy |
| Credibility eval | Blending vs raw group mean | Report with intervals |
| Leakage eval | `as_of` gate refusal | 100% blocked |
| Citation validator eval | Uncited numeric claims | Zero reach ledger |
| Calibration | Reliability curve, Brier, log score | Track over time |
| Regression tracker | Same question set across runs | Diff in evals/results/ |

## Live ledger

Scheduler polls the ledger every N seconds (default 300s). Resolver:
- Finds open questions with resolution dates that have passed
- Records outcomes
- Scores forecasts (Brier, log score, pinball)
- Writes scores to the ledger

## Technology choices

| Decision | Choice | Reason |
|----------|--------|--------|
| Language | Python 3.11+ | Stats ecosystem |
| Packaging | `uv`, `pyproject.toml` | Fast, reproducible |
| Models | Pydantic v2 | Structured outputs at every boundary |
| Storage | SQLite (WAL mode) | Simple, triggers for append-only |
| Stats | numpy, pandas, scipy, statsmodels | Standard, auditable |
| Tool layer | MCP (FastMCP) | Protocol understanding |
| LLM access | OpenAI-compatible | Swap local quantised model later |
| CLI | typer | Quick to demo |
| Tests | pytest, hypothesis | Property tests on scoring |
