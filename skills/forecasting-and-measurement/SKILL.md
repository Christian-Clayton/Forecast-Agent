---
name: forecasting-and-measurement
description: Applies statistical, actuarial, and judgmental forecasting techniques to make or evaluate predictions, quantify risk/uncertainty, or measure vague/intangible things. Use when the user asks for a forecast, probability estimate, trend extrapolation, risk quantification, capability/market/job-impact prediction, calibrated estimate, or a way to measure something that seems unmeasurable (KPIs, "how likely is X", "how big will Y get").
---

# Forecasting and Measurement

## Quick Reference (Layer 0)

**Trigger:** Need a probability, prediction, trend extrapolation, risk quantification, or a way to measure something that seems unmeasurable.
**Process in one line:** Classify domain (thin-tailed vs. fat-tailed) → if thin-tailed: base rate + decomposition + ensemble → if fat-tailed: scenarios + robustness over prediction. Always: state bottom-line number first, show arithmetic, plan to score.
**Token budget:** ~2000-4000 tokens for full application; ~500 for classification + base-rate check only.

## Overview

Forecasting and measurement are both about reducing uncertainty with disciplined method instead of gut feel. Statisticians, actuaries, superforecasters, and intelligence analysts have each built a piece of the same toolkit from different angles — actuaries price rare-but-consequential risk, statisticians extrapolate repeatable patterns, superforecasters calibrate judgment under irreducible uncertainty, and measurement theorists show that "unmeasurable" almost always means "not yet decomposed." This skill packages all four into one decision framework: classify the problem first, then apply the toolkit that actually fits it, then score yourself so the estimate improves next time.

The single biggest failure mode this skill guards against: applying precise statistical machinery (confidence intervals, regression, trend lines) to a domain that is fundamentally fat-tailed and narrative-driven, which produces false confidence rather than insight.

## Output Convention: Actuarial Style

Every answer produced with this skill follows an actuarial report's shape, not an essay's:

1. **Bottom line first.** Lead with the numeric answer — a percentage, probability, or range with a resolution date — before any narrative. A reader (or you, six months later) should get the estimate from the first line, not the last paragraph.
2. **Every claim gets a number.** Never write "likely," "probably," or "significant" without attaching a percentage or range next to it. If you genuinely cannot quantify it, say so explicitly ("no basis to quantify — judgment-only, flagged") rather than hiding behind vague language.
3. **Show the actual math, not just the technique's name.** If you invoke Bayesian updating, ensembling, credibility weighting, or a regression/trend fit, the arithmetic must appear in the working — prior → evidence/likelihood ratio → posterior; the list of estimates actually averaged; the fitted line's slope. Naming a technique without executing it is a red flag (see below), not an application of it.
4. **State the data source for every base rate.** Either cite the real dataset/publication a number came from (and how current it is), or explicitly label the number as an unverified prior/judgment call. Don't let a cited-sounding number imply data grounding that didn't happen.
5. **Working comes after**, structured as: domain classification → base rate/data source → decomposition → technique(s) applied with visible arithmetic → bias check → final calibration note.
6. **Collapse the working.** Wrap step 5's full working (everything after the bottom line) in a collapsible block (`<details><summary>Show working</summary>...</details>`) so only the bottom-line number and a one-line justification are visible by default — expand only if asked.

## When to Use

- Making or evaluating a probability/point/range forecast (markets, tech trends, timelines, job-market impact, elections, project completion dates)
- Quantifying risk or uncertainty (insurance/actuarial pricing, reserving, capital, catastrophe/tail risk)
- Designing a metric or measurement for something that feels intangible (team "quality", product "success", policy "impact")
- Reviewing someone else's forecast or metric for hidden overconfidence, bad base rates, or an ill-defined target
- Building a personal or team track record of calibrated predictions

**When NOT to use:** Routine descriptive statistics with no predictive claim (a report of what already happened), or decisions where the outcome is already known/deterministic.

## Step 0: Classify Before You Model

Two gating questions come before any technique. Skipping them is the most common way forecasts go wrong.

**Q1 — Is the domain thin-tailed or fat-tailed?** (Taleb's Mediocristan vs. Extremistan)
- **Thin-tailed / Mediocristan:** outcomes cluster, no single event dominates the average (height, most operational metrics, well-behaved demand series, mortality in a stable population). → Use **Toolkit A**.
- **Fat-tailed / Extremistan:** one extreme event can dwarf everything before it (markets, wars, pandemics, viral tech adoption, breakthrough capability jumps). → Use **Toolkit B**. Precision-forecasting techniques here produce *false* confidence, not accuracy.
- Many real questions are mixed (e.g., "AI job-market impact" has thin-tailed sub-components — attrition rates, hiring volume — nested inside a fat-tailed whole — a single model breakthrough). Decompose first, classify each piece separately.

**Q2 — What, exactly, is being measured or forecast?** (Hubbard's clarification step)
Before picking a method, restate the question until it is decomposable and falsifiable: "Will AI be a big deal for jobs" is not answerable; "what fraction of UK job postings in occupation X will list an LLM-adjacent skill by date Y" is. If the target can't be stated as something that will observably resolve, fix that before doing any modeling — the measurement/forecasting technique cannot rescue an ill-defined question.

## Toolkit A — Thin-Tailed / Repeatable Domains

Use base rates, decomposition, and historical pattern-extrapolation; track calibration.

| Technique | What it does | Source discipline |
|---|---|---|
| **Outside view / reference class forecasting** | Start from the base rate of similar past cases before adjusting for this case's specifics — corrects the planning fallacy (specific-case narratives are systematically over-optimistic) | Tetlock; Kahneman & Tversky; Flyvbjerg |
| **Fermi decomposition** | Break one fuzzy question into smaller, independently-estimable sub-questions, then recombine | Tetlock; physics estimation tradition |
| **Naive/seasonal-naive baselines first** | Before fitting anything complex, forecast with "same as last period" or "same as last year" — M-competition results (Makridakis) repeatedly show simple methods are hard to beat and any complex model must clear this bar to be worth using | Time-series forecasting (Hyndman & Athanasopoulos) |
| **STL / classical decomposition** | Split a series into trend + seasonality + residual before modeling each separately | Statistics |
| **Exponential smoothing (ETS) / ARIMA** | Standard workhorse models for repeatable series with trend/seasonality and autocorrelation | Statistics |
| **State-space models / Kalman filtering** | Model a system with an unobserved evolving state (e.g., "true" demand) plus noisy observations; updates recursively as data arrives — mechanistic cousin of Bayesian updating | Statistics / engineering |
| **Bayesian updating** | Start with a prior probability, update it incrementally as evidence arrives, rather than re-deriving from scratch each time — mirrors superforecasters' "small, frequent revisions" habit | Statistics |
| **Credibility theory (Bühlmann credibility)** | Actuarial technique for blending an individual entity's own experience with the wider class average, weighted by how much data the individual has — the actuarial name for Bayesian shrinkage, used for pricing when an individual risk has too little history to trust alone | Actuarial science |
| **GLMs (frequency–severity modeling)** | Model event frequency and event severity as separate generalized linear models, then combine — the standard actuarial pricing technique, more robust than a single regression on total loss | Actuarial science |
| **Loss reserving (chain-ladder, Bornhuetter-Ferguson)** | Project ultimate losses from a triangle of historical claims development — a template for "how much of this trend has yet to show up" problems generally | Actuarial science |
| **Ensembling / combining forecasts** | Average multiple independent forecasts/models — the combined forecast is empirically almost always more accurate than any single component (Bates & Granger result), and is the mechanical version of "wisdom of crowds" | Statistics |
| **Ensemble spread as signal** | When multiple models/approaches disagree, don't just average — report the spread explicitly as a direct confidence indicator. Tight spread = high confidence in the average; wide spread = the situation is genuinely unpredictable and you should switch to Toolkit B (robustness over prediction). From modern numerical weather prediction (ECMWF ensemble system, 2020s): ensemble spread is itself a forecast of uncertainty, not noise to average out | Meteorology / ECMWF |
| **Cross-validation via walk-forward (not random split)** | For any time-dependent data, validate by training on the past and testing on the future in rolling windows — random k-fold validation leaks future information and overstates accuracy | Statistics |

## Toolkit B — Fat-Tailed / Novel Domains

Precision forecasting techniques break down here. Shift the goal from "predict the number" to "bound the uncertainty and stay robust to whichever tail hits."

| Technique | What it does | Source discipline |
|---|---|---|
| **Extreme value theory (EVT)** | Models the tail of a distribution directly (block maxima / peaks-over-threshold) rather than fitting a single distribution to the whole range — the actuarial/catastrophe-modeling standard for rare, high-impact events | Actuarial science / statistics |
| **Robustness over prediction (barbell, optionality)** | Structure exposure so you survive or benefit regardless of which extreme hits, instead of betting on a specific outcome — cap the downside, leave the upside open | Taleb |
| **Scenario planning** | Build 3–4 internally-consistent, structurally-different futures (not one predicted future) and stress-test decisions against all of them | Schwartz; Institute for the Future |
| **Delphi method** | Anonymous, iterative expert polling with controlled feedback between rounds, used specifically when historical data is too thin for statistical methods | Foresight/futures studies |
| **Cross-impact analysis** | Assess how the occurrence of one forecast event would shift the probability of others, instead of forecasting each in isolation | Foresight/futures studies |
| **Structured Analytic Techniques / Analysis of Competing Hypotheses (ACH)** | List every plausible hypothesis explicitly, then disconfirm rather than confirm — built for the intelligence-analysis problem of sparse, ambiguous evidence, and a strong corrective for narrative fallacy | Heuer, intelligence analysis tradecraft |
| **Premortem** | Before committing to a forecast/plan, imagine it has already failed and work backward to explain why — surfaces risks that forward reasoning misses | Gary Klein |
| **Prediction markets / wisdom of crowds** | Aggregate many independent, incentivized judgments — works when errors are independent and non-correlated; fails when everyone shares the same blind spot | Behavioral economics |
| **S-curve / diffusion-of-innovation trend extrapolation** | Model technology adoption or capability growth as a logistic (S-shaped) curve rather than linear extrapolation — most real growth trajectories saturate | Rogers; Gartner Hype Cycle |
| **Generalist/cross-domain analogy** | For genuinely novel domains, deliberately borrow structural analogies from unrelated fields rather than over-trusting the (possibly already-stale) specialist literature of the field itself | Epstein |

## Measurement Toolkit (for "how do I measure X")

Hubbard's core claim, well-supported in practice: anything that matters is measurable, because "measurement" only requires reducing uncertainty, not eliminating it.

1. **Decompose the vague thing** into observable proxies (e.g., "team quality" → defect escape rate, cycle time, on-call incident count) rather than declaring it immeasurable.
2. **Estimate in calibrated ranges, not points.** State a 90% confidence interval you'd genuinely bet on, not a single number — this is trainable (calibration exercises: answer trivia with a range you're 90% sure contains the true answer, check your actual hit rate; well-calibrated people are right ~90% of the time, not 99% or 60%).
3. **Compute the value of additional information before gathering more data.** If a decision doesn't change regardless of the measurement's outcome, don't spend budget refining it further.
4. **Check construct validity and reliability** of any metric: does it actually track the underlying thing (validity), and does it produce consistent results on repeated measurement (reliability)? A precise-looking number that measures the wrong construct is worse than a rough one that measures the right one.
5. **Watch for Goodhart's Law:** "when a measure becomes a target, it ceases to be a good measure" — any metric used to evaluate or incentivize people will eventually be gamed; prefer a small basket of metrics that are hard to game simultaneously over one clean KPI.

## Calibration and Scoring — Closing the Loop

A forecast without a scoring mechanism never improves. Score every forecast that resolves.

| Scoring rule | Use for | Notes |
|---|---|---|
| **Brier score** | Binary/categorical probability forecasts | Mean squared error between forecast probability and outcome (0/1); lower is better; the standard superforecasting metric |
| **Log score** | Binary/categorical probability forecasts | Penalizes confident wrong answers much more harshly than Brier — better when overconfidence is the main risk |
| **Pinball loss / CRPS** | Quantile or range forecasts | Scores whether stated intervals actually contain outcomes at the stated rate |

**Practical loop:** write the forecast down with an explicit probability/range and a resolution date → do nothing until it resolves → score it → review misses for *why* (bad base rate? wrong toolkit? cognitive bias?) → adjust the process, not just the number. Building this loop publicly (Good Judgment Open, Metaculus, or a personal log) is the closest thing this field has to a credential — track record beats certification here.

## Cognitive Bias Guardrails

| Bias | How it distorts a forecast | Corrective technique |
|---|---|---|
| Overconfidence | Intervals too narrow; point estimates stated as certainties | Calibration training; state ranges; pre-register a probability |
| Anchoring | First number seen (yours or someone else's) pulls the final estimate toward it | Generate an independent estimate before seeing others' numbers, then reconcile |
| Base-rate neglect / planning fallacy | Vivid specific-case narrative crowds out the outside view | Reference class forecasting — find the base rate first |
| Narrative fallacy | A coherent story feels more probable than a boring one, even when it isn't | ACH — force-list disconfirming hypotheses, not just the appealing one |
| Availability | Recent or memorable events treated as more likely than base rates support | Check the actual historical frequency, not recall ease |
| Hindsight bias | Past outcomes feel like they were obviously predictable after the fact | Compare against your own logged, timestamped pre-outcome forecast |

## LLM-Specific Failure Modes

| Failure Mode | How This Skill Guards Against It |
|---|---|
| **Technique name-dropping without execution** | Explicit requirement to show arithmetic, not just cite method names |
| **False precision on fat-tailed domains** | Domain classification gate prevents applying confidence intervals where they're meaningless |
| **Ungrounded base rates** | Every base rate must cite its data source or be explicitly flagged as judgment |
| **Linear extrapolation of S-curves** | S-curve/diffusion model explicitly called out as default for adoption/growth phenomena |
| **Unfalsifiable hedging** | Resolution date and scoring plan required — prevents producing forecasts that can never be wrong because they're never checked |

## Effectiveness Signal

This skill worked if: (a) the forecast resolves and scores well (Brier <0.25 for binary, or within stated confidence interval); (b) the domain classification prevented applying the wrong toolkit; (c) a naive baseline was beaten by the chosen method.

## Red Flags

- A single point-estimate forecast with no stated uncertainty or confidence interval
- Precision-looking statistical output (regression, ARIMA, confidence intervals) applied to a domain not first checked for fat tails
- A technique named ("Bayesian updating", "ensembling") with no actual arithmetic shown — name-dropping instead of applying
- A base-rate number with no stated data source, presented as if it were verified
- A metric adopted as a target with no check for how it could be gamed (Goodhart's Law)
- "Expert judgment" presented with high confidence but no track record or calibration history behind it
- A forecast that can't be falsified — no resolution date, no observable outcome
- Extrapolating a trend linearly when the underlying process is adoption/growth-shaped (should be S-curve)
- Validating a time-series model with random train/test splits instead of walk-forward splits
- No plan to score the forecast once it resolves

## Verification (Read BEFORE Starting and After Completing)

- [ ] Domain classified as thin-tailed or fat-tailed (or decomposed into sub-parts that are)
- [ ] The forecasting question is stated in a falsifiable, resolvable form
- [ ] Bottom-line percentage/range is stated first, before the working
- [ ] Every qualitative claim ("likely", "significant") has a number attached, or is explicitly flagged as unquantified judgment
- [ ] Toolkit chosen matches the classification (A for repeatable, B for novel/fat-tailed)
- [ ] Base rate / reference class considered before adjusting for case specifics, with its data source stated
- [ ] Any named quantitative technique (Bayesian update, ensembling, regression) has its actual arithmetic shown, not just its name cited
- [ ] When multiple models/forecasters are ensembled, the spread is reported as a confidence signal — wide disagreement triggers a switch to Toolkit B rather than blind averaging
- [ ] A scoring plan exists (Brier/log score/CRPS) for when the forecast resolves
- [ ] Any metric involved has been checked for Goodhart's-Law gameability

## Chaining Protocol

| After completing this skill, check... | Then invoke... |
|---|---|
| The forecast feeds a decision between options | `decision-making-under-uncertainty` to convert probability into action |
| The domain was classified as fat-tailed / deep uncertainty | `scenario-planning` instead of single-point forecasting |
| A past forecast missed badly and you need to understand why | `causal-reasoning-and-root-cause-analysis` for the miss mechanism |
| The base rates or evidence feeding the forecast need vetting | `critical-thinking-and-argument-evaluation` first |
| You only need a rough order-of-magnitude number, not a calibrated forecast | `fermi-estimation` (cheaper, faster) |
| Ensemble members disagree widely (spread > 2x the mean) | Switch to Toolkit B — treat the spread as a signal of genuine unpredictability, not noise to average |
