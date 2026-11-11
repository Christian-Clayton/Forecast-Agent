---
name: fermi-estimation
description: Produces a fast, defensible order-of-magnitude estimate for a quantity with no direct data, by decomposing it into a chain of smaller estimable factors — the physicist's "how many piano tuners are in Chicago" technique. Use whenever the user needs a quick sanity-check number, a feasibility gut-check, a rough sizing (market size, cost, time, headcount, resource requirement) with no time or data for rigorous measurement, or wants to sanity-check a number someone else produced.
---

# Fermi Estimation

## Quick Reference (Layer 0)

**Trigger:** Need an order-of-magnitude number fast, with no data available to look up. Or sanity-checking whether a claimed number is in the right ballpark.
**Process in one line:** Decompose into 3-5 estimable factors → anchor each factor → multiply through → sanity-check against a known comparable → state as a range, not a point.
**Token budget:** ~400-800 tokens. This is deliberately the cheapest analytical skill.

## Overview

`forecasting-and-measurement` already holds that "unmeasurable" almost always means "not yet decomposed" — Fermi estimation is the fastest, sharpest form of that decomposition, purpose-built for a single order-of-magnitude number needed right now, not a calibrated probability or a tracked trend. It's also frequently the number a `decision-making-under-uncertainty` tree needs as an input and doesn't have time to look up properly. Use this when speed matters more than precision, and precision would be false anyway given the total absence of data.

## Output Convention

Lead with the final estimate **and its plausible order-of-magnitude range** (e.g., "~4,000, plausible range 1,000–15,000"), never a falsely precise point number. Then show the decomposition chain. Wrap the decomposition in a collapsible block (`<details><summary>Show working</summary>...</details>`) so only the estimate and range are visible by default — expand only if asked.

## When to Use

- No data or time exists to look the number up properly, but a decision or gut-check needs one now
- Sanity-checking whether a claimed number (a vendor quote, a projected growth figure, a headline statistic) is even in the right ballpark
- A feasibility check before investing in rigorous measurement — is this even worth measuring precisely?

**When NOT to use:** the real number is actually cheap and available to look up (look it up, don't estimate it); the decision is high-stakes enough to justify `forecasting-and-measurement`'s fuller calibrated toolkit.

## Step 1: Decompose Into a Chain of Estimable Factors

Break the target quantity into a short chain of factors, each individually far easier to estimate than the whole. Classic example — piano tuners in Chicago: population → households → fraction owning a piano → tunings per piano per year → tunings one tuner can do per year → divide through.

## Step 2: Anchor Each Factor Fast

For each factor, use whichever anchor is fastest: a known reference point (a well-known population, price, or rate), a personal-experience anchor, or a plausible round number. Don't spend more than a few seconds per factor — speed is the entire point of the technique.

## Step 3: Multiply Through and Sanity-Check

Multiply the chain, then sanity-check the resulting order of magnitude against any known comparable reference (is this bigger or smaller than a similar known quantity by a factor that actually makes sense?).

## Step 4: State an Honest Range, Not False Precision

A well-built Fermi estimate is typically right to within a factor of 2–3 of the true value — that's the actual claim being made. State it as a range, not a specific digit count.

## Step 5 (For Consequential Estimates): Cross-Check With a Second Chain

Redo the decomposition through a genuinely different chain of factors (e.g., estimate piano tuners from tuner-business directory density instead of demand-side ownership rates). If two independent chains land in the same order of magnitude, confidence rises substantially. If they diverge by more than ~10x, at least one chain has a flawed factor — find it before trusting either.

## Common Anchor Reference Points

| Category | Useful anchors |
|---|---|
| Population | World ~8B, US ~335M, major world city ~1–10M, mid-size city ~100k–1M |
| Household size | ~2.5 people/household (developed economies), higher in many developing economies |
| Time | Working year ≈ 2,000 hours; a person's career ≈ 40 years |
| Money | Rough global/national GDP figures, typical salary bands by role/region |

## LLM-Specific Failure Modes

| Failure Mode | How This Skill Guards Against It |
|---|---|
| **False precision from training data** | Explicit range requirement prevents reporting a memorized-seeming number with unjustified decimal places |
| **Anchoring on the first number that comes to mind** | Decomposition into factors forces deriving the answer rather than pattern-matching a recalled figure |
| **Confabulated reference points** | Cross-check step (second independent chain) catches when a single chain's anchor was wrong |

## Effectiveness Signal

This skill worked if: (a) the estimate is within a factor of 3 of the true value (the actual claim a good Fermi estimate makes); (b) two independent decomposition chains converge; (c) the estimate was fast enough to preserve the token budget for the downstream decision.

## Red Flags

- Reporting a Fermi estimate with false precision ("4,127" instead of "~4,000")
- A single decomposition chain with no cross-check for an estimate that's actually consequential
- A chain with more than ~5–6 multiplied factors, each compounding uncertainty — prefer a shorter chain of higher-confidence factors
- Treating a Fermi estimate as a substitute for real measurement once the stakes actually justify it

## Verification (Read BEFORE Starting and After Completing)

- [ ] Quantity decomposed into a short chain (ideally 3–5) of independently estimable factors
- [ ] Each factor anchored to something concrete, not a pure guess
- [ ] Final answer stated as an order-of-magnitude range, not a false-precision point number
- [ ] For consequential estimates, a second independent decomposition chain cross-checks the first
- [ ] Not used as a substitute for real data that was actually cheap and fast to obtain directly

## Chaining Protocol

| After completing this skill, check... | Then invoke... |
|---|---|
| The estimate feeds a consequential decision and deserves more precision | `forecasting-and-measurement` for a calibrated, tracked forecast |
| The estimate is an input to a decision tree | `decision-making-under-uncertainty` to use it as a probability/payoff input |
| The number seems surprisingly large/small and needs sanity-checking against evidence | `critical-thinking-and-argument-evaluation` to vet the result |
| Two independent chains diverge by >10x | One chain has a flawed factor — diagnose which before trusting either |
