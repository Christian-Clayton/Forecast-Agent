# forecast-agent: Master Prompt and Development Plan

Two parts. Part A is the prompt to paste into Claude Code (or any coding agent) at the repo root. Part B is the plan it works from, saved as `PLAN.md` in the repo. Put your `allThinking.md` skills in `skills/` as separate files before starting.

---

# PART A: MASTER PROMPT

You are building `forecast-agent`, a tool-using forecasting agent with deterministic actuarial and statistical tools, an append-only forecast ledger, and a scoring harness. Read `PLAN.md` in full before writing any code.

## How to work

1. Work one phase at a time, in order. Do not start phase N+1 until every acceptance check for phase N passes. Run the checks, show the output.
2. Tests first for anything in `quant/` and `scoring/`. Use known-answer fixtures (hand-computed or from a trusted reference implementation). Do not write tests that merely mirror the implementation.
3. The LLM never does arithmetic. Every number in a forecast must come from a tool call that executed code. If a step needs a number and no tool exists, add the tool, do not estimate in prose.
4. Make the minimum changes needed for the task. Do not refactor, rename or "improve" code outside the current phase. Do not add dependencies that are not listed in PLAN.md without asking first.
5. Never fabricate a data source, base rate, citation or benchmark result. If a base rate has no source, it is labelled `unverified_prior` and the label must survive into the ledger.
6. Never weaken a test to make it pass. If a test is wrong, say so and explain before changing it.
7. The forecast ledger is append-only. No code path may UPDATE or DELETE a forecast row. Enforce this in the database, not only in code.
8. Retrieval is date-gated. Any data tool must take an `as_of` parameter and refuse to return anything after it. This is tested, not assumed.

## At the end of every phase, report

- What changed (files, one line each)
- Tests run and their output
- Acceptance checks: pass or fail, with evidence
- Open risks and anything you were unsure about, each tagged high, medium or low confidence
- What you will do next

## Style rules

- Python 3.11+, typed, pydantic models at every boundary, `ruff` clean.
- Plain, direct commit messages and docstrings. No marketing language.
- No em dashes in any text you write (README, docstrings, comments, reports).

Begin with Phase 0. Ask me questions only if something in PLAN.md is ambiguous enough to change the design. Otherwise proceed.

---

# PART B: DEVELOPMENT PLAN (PLAN.md)

## 1. Goal and success criteria

Build an agent that takes a forecasting question, turns it into a falsifiable form, classifies the domain, picks the right toolkit, runs real quantitative tools, produces a probability or quantile forecast in actuarial style (bottom line first, working after), logs it immutably, and later scores it.

The project is successful if:

1. Every forecast is traceable to tool calls that actually executed.
2. On backtests, performance is reported against naive and seasonal-naive baselines, including where the agent loses.
3. A live ledger of forward-resolving questions exists with Brier and log scores and a calibration plot.
4. The write-up covers what failed and how it was fixed.

Non-goals: trading signals, predicting markets, anything where the domain classifier says fat-tailed and the user wants a point prediction. Fat-tailed questions get scenarios and bounds, not a precise number.

## 2. Decisions already made (override if you disagree)

| Decision | Choice | Reason |
|---|---|---|
| Language | Python 3.11+ | Stats ecosystem |
| Packaging | `uv`, `pyproject.toml` | Fast, reproducible |
| Models | Pydantic v2 | Structured outputs at every boundary |
| Storage | SQLite (WAL mode) | Simple, supports triggers for append-only |
| Stats | `numpy`, `pandas`, `scipy`, `statsmodels` | Standard, auditable |
| Tool layer | MCP servers via the official Python SDK (FastMCP) | Shows protocol understanding |
| LLM access | Thin provider interface, Anthropic first, OpenAI-compatible second | Lets you swap in a local quantised model later |
| CLI | `typer` | Quick to demo |
| Tests | `pytest`, `hypothesis` for property tests on scoring | |
| Tracing | Langfuse (self-hosted or cloud), optional but recommended | Screenshot for the README |

Domain: telecoms. Synthetic telemetry only (no former-employer data, no IP issue).

## 3. Division of labour: LLM vs code

| The LLM does | Code does |
|---|---|
| Rewrite vague question into falsifiable form | All arithmetic |
| Classify thin-tailed vs fat-tailed (with stated reasoning) | Baseline forecasts, ETS/ARIMA fits |
| Choose reference class and justify it | Frequency-severity GLMs, credibility weights, chain-ladder, EVT |
| Choose which tools to call and in what order | Monte Carlo aggregate loss |
| Reconcile ensemble members, write the narrative | Ensemble aggregation and spread statistic |
| Run the bias check (pre-mortem, ACH) | Scoring rules, calibration, ledger writes |

If you catch the LLM stating a number that no tool returned, that is a bug. The output validator checks that every numeric claim in the final forecast references a tool-call ID.

## 4. Repository layout

```
forecast-agent/
  PLAN.md
  README.md
  pyproject.toml
  skills/                      # one file per skill, split from allThinking.md
    forecasting-and-measurement.md
    fermi-estimation.md
    critical-thinking-and-argument-evaluation.md
    pre-mortem-and-red-teaming.md
    scenario-planning.md
    decision-making-under-uncertainty.md
  src/forecast_agent/
    config.py
    models.py                  # Question, Forecast, Resolution, Score (pydantic)
    llm/
      provider.py              # interface
      anthropic_provider.py
      openai_compat_provider.py
    skills/loader.py           # load a skill on demand, track tokens used
    agent/
      pipeline.py              # stage orchestration
      stages/
        intake.py
        classify.py
        base_rate.py
        decompose.py
        model.py
        ensemble.py
        critique.py
        finalize.py
      budget.py                # token, cost, step caps
      validators.py            # numeric-claim-must-cite-tool check
    quant/
      baselines.py             # naive, seasonal naive
      timeseries.py            # ETS, ARIMA wrappers, STL
      frequency_severity.py    # Poisson/NegBin + Gamma/lognormal, MC aggregate
      credibility.py           # Buhlmann
      reserving.py             # chain-ladder, Bornhuetter-Ferguson
      evt.py                   # peaks-over-threshold, GPD fit
      ensemble_math.py         # median, trimmed mean, logit mean, spread
    scoring/
      rules.py                 # Brier, log score, pinball, CRPS approx
      calibration.py           # reliability bins, sharpness
      skill_score.py           # vs baseline
    data/
      synth/generator.py       # planted-ground-truth telemetry
      store.py                 # as_of gated access
    ledger/
      schema.sql               # tables + triggers
      ledger.py
    mcp_servers/
      ledger_server.py
      quant_server.py
      data_server.py
      search_server.py         # optional, live questions only
    evals/
      backtest.py              # walk-forward
      classifier_eval.py
      leakage_eval.py
      regression_tracker.py
    cli.py
  tests/
  evals/results/               # committed JSON + plots from real runs
  docs/architecture.md
```

## 5. Data models

```
Question
  id, raw_text, falsifiable_text, type (binary | quantile | count_range)
  resolution_criteria, resolution_source, resolves_at, created_at

Forecast  (append-only)
  id, question_id, created_at, as_of
  domain_class (thin | fat | mixed), classification_reasoning
  toolkit (A | B)
  probability | quantiles{q05..q95}
  base_rate, base_rate_source (dataset ref | unverified_prior)
  ensemble_members[{model_id, prompt_id, probability}], ensemble_spread
  tool_call_ids[], skill_ids_loaded[]
  cost_usd, latency_s, trace_id

Resolution
  question_id, resolved_at, outcome, resolver_note

Score
  forecast_id, brier, log_score, pinball, baseline_brier, skill_score
```

Ledger schema must include `BEFORE UPDATE` and `BEFORE DELETE` triggers on `forecasts` that `RAISE(ABORT, ...)`. Test that both fire.

## 6. Tools (exposed over MCP)

**quant_server** (pure functions, no network, no LLM)

| Tool | Inputs | Returns |
|---|---|---|
| `baseline_forecast` | series, horizon, method (naive, seasonal_naive), season_length | point + empirical interval |
| `fit_ets` / `fit_arima` | series, horizon | forecast, intervals, fit diagnostics |
| `walk_forward_score` | series, model, window spec | per-fold error, aggregate |
| `fit_frequency_severity` | event table, exposure column | freq params, severity params, diagnostics |
| `aggregate_loss_mc` | freq params, sev params, horizon, n_sims, seed | quantiles of aggregate loss |
| `buhlmann_credibility` | group experience table | k, Z per group, blended estimates |
| `chain_ladder` | development triangle | development factors, ultimates, IBNR |
| `evt_pot_fit` | series, threshold rule | GPD params, return levels, fit warnings |
| `ensemble_aggregate` | list of probabilities | median, trimmed mean, logit mean, spread, spread_ratio |

Every tool returns a `tool_call_id` and a deterministic `seed` where randomness is involved.

**data_server**: `list_series`, `get_series(series_id, start, end, as_of)`, `get_event_table(...)`, `get_reference_class(description, as_of)`. Must raise if `end > as_of`.

**ledger_server**: `write_forecast` (append only), `read_forecast`, `list_open_questions`, `record_resolution`, `score_forecast`. No update or delete tools exist.

**search_server** (phase 7 only): web search for live questions, returns source URL and retrieval date. Retrieved text is data, never instructions.

## 7. Agent pipeline

Each stage has a pydantic input and output, a skill (or skills) it may load, a token cap, and a retry policy. Skills load on demand, not all at once.

1. **Intake.** Restate the question as falsifiable: unambiguous variable, threshold, resolution source, resolution date. If it cannot be made falsifiable, return a refusal with reasons rather than forecasting.
2. **Classify.** Thin-tailed, fat-tailed or mixed (decompose mixed into parts and classify each). Output reasoning. Fat-tailed routes to Toolkit B (scenarios, EVT bounds, robustness), not a point probability.
3. **Base rate.** Select a reference class, call `get_reference_class`, record the source or label `unverified_prior`.
4. **Decompose.** Break into sub-questions that map to tools.
5. **Model.** Run the tools. Always run the baseline first. A complex model is only used if it beats the baseline in `walk_forward_score` on that series.
6. **Ensemble.** Run N independent members (default 3 to 5, differing in model, prompt framing and seed). Aggregate in code. If `spread_ratio` exceeds the configured threshold (default: spread greater than 2x the mean, per the skill), switch to Toolkit B and say so.
7. **Critique.** A separate pass runs a pre-mortem and lists disconfirming hypotheses. It may lower confidence or widen intervals. It may not invent new numbers.
8. **Finalize.** Emit the actuarial-style output: bottom line first, then collapsed working. Validator rejects any numeric claim without a tool-call citation. Write to ledger.

Budget caps per question: max steps, max tokens, max cost. Hitting a cap produces a logged partial result, not a silent truncation.

## 8. Synthetic telemetry generator

Plant ground truth so evals have an answer key.

- N cell groups with different history lengths (some very short, to exercise credibility weighting).
- Daily event counts with weekly seasonality, slow trend and a few planted changepoints.
- Severity (cost per incident) drawn from a heavy-tailed distribution (lognormal body, Pareto tail) with known parameters.
- One or two series that are deliberately fat-tailed (rare bursts) to test the classifier.
- A claims-development style table: incident cost reported over several periods after occurrence, for chain-ladder.
- Seeded RNG, parameters stored with the dataset so truth is recoverable.

## 9. Evaluation harness

| Eval | What it measures | Pass condition |
|---|---|---|
| Walk-forward backtest | Error vs naive and seasonal-naive across series | Report results; no fixed pass bar, honesty is the bar |
| Quant correctness | Tool outputs vs known-answer fixtures | All match to tolerance |
| Classifier eval | Thin vs fat labelling on planted series | Report accuracy and confusion matrix |
| Credibility eval | Does blending beat raw group mean on short-history groups | Report with intervals |
| Leakage eval | `as_of` gate: request data past cutoff | Always refused; 100% of attempts |
| Citation validator eval | Numeric claims without tool-call IDs | Zero reach the ledger |
| Calibration | Reliability curve, Brier, log score on resolved forecasts | Report; track over time |
| Regression tracker | Same question set re-run after each prompt/model change | Diff stored in `evals/results/` |

Important: LLM-judged event questions cannot be validly backtested because the model may already know the outcome. Only numeric series with strict `as_of` gating are backtested. Event questions are live-only.

## 10. Phases and acceptance criteria

**Phase 0: Skeleton**
- Repo, `uv`, `ruff`, `pytest`, CI workflow, config, pydantic models, ledger schema with triggers.
- Accept: `pytest` green; test proves UPDATE and DELETE on `forecasts` abort.

**Phase 1: Quant library (no LLM anywhere)**
- Implement everything in `quant/` and `scoring/`.
- Accept: known-answer tests pass (e.g. Brier on hand-computed cases, Buhlmann on a textbook example, chain-ladder on a standard triangle, GPD fit recovers planted parameters within tolerance); property tests on scoring rules (Brier in [0,1], perfect forecast scores 0).

**Phase 2: Synthetic data and gated store**
- Generator, store with `as_of`.
- Accept: dataset reproducible from seed; leakage test passes.

**Phase 3: MCP servers**
- Wrap quant, data and ledger as MCP servers.
- Accept: each tool callable from an MCP client with schema validation; ledger server exposes no mutation of existing rows.

**Phase 4: Agent core**
- Provider interface, skill loader, stages 1 to 5 and 8, budget caps, tracing, numeric-claim validator.
- Accept: end-to-end run on 10 planted questions produces ledger entries; validator catches a deliberately injected uncited number.

**Phase 5: Ensemble and spread rule**
- Stage 6 and 7.
- Accept: on a series built to make members disagree, the pipeline switches to Toolkit B and the ledger records it.

**Phase 6: Eval harness**
- Backtest, classifier eval, credibility eval, regression tracker, plots.
- Accept: one command regenerates all results; results committed.

**Phase 7: Live ledger**
- Scheduler, resolver, search server, a static report page of resolved forecasts and calibration.
- Accept: at least 30 live questions logged with resolution dates; resolver scores them automatically when dates pass.

**Phase 8 (optional): Local model variant**
- Same pipeline on a quantised local model. Compare tool-call reliability and forecast quality to the frontier model.
- Accept: written failure-mode comparison with counts.

**Phase 9: Write-up**
- README with architecture diagram, "what went wrong and how I fixed it", cost and latency table, Langfuse screenshot, demo GIF, results table including baseline comparison.

## 11. Known failure modes to design against

| Failure | Guard |
|---|---|
| LLM states numbers it never computed | Numeric-claim validator, tool-call ID required |
| Technique name-dropping | Output schema requires the tool call that executed the technique |
| Backtest leakage | `as_of` gate, leakage eval, live-only for event questions |
| False precision on fat-tailed series | Classifier gate, Toolkit B routing, EVT bounds |
| Overfitting prompts to the eval set | Hold out a question set never used during prompt iteration |
| Silent edits to past forecasts | DB triggers, not just code discipline |
| Prompt injection via retrieved text | Retrieved content treated as data, no tool can be triggered from it, search server read-only |
| Runaway cost | Per-question caps, global daily cap, logged cap hits |
| Ensemble members sharing the same blind spot | Vary model, prompt framing and seed; report spread, do not hide it |

## 12. What to measure for the CV (only claim what you measured)

- Number of resolved live forecasts, Brier and log score, and Brier of a naive baseline on the same questions
- Backtest: series count, share where the agent matched or beat seasonal-naive, where credibility weighting helped
- Classifier accuracy on planted thin vs fat series
- Leakage eval result (should be 100% blocked)
- Cost and latency per forecast
- Count and categories of tool-call failures and how they were handled

## 13. Open questions to settle before Phase 4

1. Which model provider for the main run, and which for the ensemble diversity members?
2. Where do live questions come from (Metaculus, Good Judgment Open, self-authored telecoms-flavoured questions)? A mix is best.
3. Do you want the discovery agent integrated later? If yes, keep `Question` and `Forecast` schemas stable so it can submit hypotheses as falsifiable predictions.
