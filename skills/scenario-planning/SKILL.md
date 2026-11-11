---
name: scenario-planning
description: Builds several structurally distinct, internally consistent plausible futures (not one forecast) to stress-test a strategy against deep uncertainty — Shell's and RAND's scenario-planning methodology. Use when the uncertainty is too deep or structural for a calibrated probability forecast to grip onto (new markets, geopolitical shifts, technology disruption, multi-decade strategy), or when the user wants to pressure-test a plan against multiple different futures rather than the single most-likely one.
---

# Scenario Planning

## Quick Reference (Layer 0)

**Trigger:** Uncertainty is too deep/structural for a calibrated probability forecast; need to stress-test a strategy against multiple possible futures.
**Process in one line:** Identify driving forces → pick the 2 most critical uncertainties (high-impact AND genuinely unpredictable) → build 2×2 matrix of structurally distinct scenarios → stress-test strategy → identify no-regrets moves + leading indicators.
**Token budget:** ~2000-3500 tokens for full application; ~800 for axes + scenario names only.

## Overview

`forecasting-and-measurement`'s calibrated-probability approach works when uncertainty is "known unknowns" with enough of a track record to score against. Scenario planning is the tool for deep, structural uncertainty where a single-point probability forecast would be false precision — RAND and Shell built this method specifically because probabilistic forecasting fails past a certain time horizon or uncertainty depth. Its output — several distinct futures — then feeds `decision-making-under-uncertainty`: the goal becomes a strategy that's robust across scenarios, not optimal in only one.

## Output Convention

Lead with the two chosen critical-uncertainty axes and the resulting scenario names/one-line summaries, then the strategy implication (what's robust across all scenarios vs. what only works in one). Show the full scenario narratives in a collapsible block (`<details><summary>Show working</summary>...</details>`) — expand only if asked.

## When to Use

- The uncertainty is structural/deep (new technology, geopolitical shift, multi-decade strategy) — too deep for `forecasting-and-measurement`'s calibrated toolkit to have a track record to grip
- A strategy needs pressure-testing against more than the single most-likely future
- Stakeholders disagree sharply about "what's going to happen" — scenario planning routes around the disagreement by building several futures instead of arguing over one

**When NOT to use:** the uncertainty is shallow/precedented enough for a real calibrated forecast (use `forecasting-and-measurement` instead); a single decision needs one expected-value answer, not a set of futures (use `decision-making-under-uncertainty`).

## Step 1: Identify Driving Forces

List everything that could plausibly affect the outcome — both **predetermined** elements (near-certain, e.g. demographic trends already baked in) and genuinely **uncertain** ones.

## Step 2: Select the Two Most Critical Uncertainties

From the uncertain list, pick the two forces that are simultaneously high-impact on the outcome *and* genuinely unpredictable (not resolvable by more research). This is the single most important step, and the one most scenario exercises get wrong — by picking axes that are important but not actually uncertain, or uncertain but not actually important.

## Step 3: Build the 2×2 Matrix

Cross the two axes to produce four scenarios. Each quadrant must be a genuinely different world, not four points on the same optimistic-to-pessimistic spectrum of one story.

## Step 4: Name and Narrate Each Scenario

Give each scenario a memorable name and an internally consistent narrative: what led here, what the world looks like — and critically, the early **leading indicators** that would signal which scenario is actually unfolding. Attaching trip-wires to each scenario is what separates this from four disconnected guesses.

## Step 5: Stress-Test the Strategy

Check the current strategy against all four scenarios, not just the expected one:

- **No-regrets moves** — robust across all four scenarios; do these regardless
- **Scenario-specific bets** — only pay off in one scenario; flag explicitly as a bet and size accordingly
- **Optionality-preserving moves** — defer commitment until an early indicator resolves which scenario is unfolding

## Step 6: Monitor the Leading Indicators

Set up monitoring against the Step 4 indicators so it's possible to tell early which scenario is actually happening, rather than only recognizing it after the fact.

## LLM-Specific Failure Modes

| Failure Mode | How This Skill Guards Against It |
|---|---|
| **Optimism/pessimism spectrum disguised as scenarios** | Explicit requirement that scenarios are structurally distinct worlds, not variants of one story |
| **Collapsing back to a single forecast** | Red flag against assigning probabilities and averaging — that defeats the method's purpose |
| **Anchoring on the "most likely" scenario** | Equal treatment requirement: strategy must be tested against all four, not just the expected one |
| **Vague scenario narratives** | Leading-indicator requirement forces concrete, observable signals for each scenario |

## Effectiveness Signal

This skill worked if: (a) the four scenarios are genuinely structurally distinct (not just degree-variations); (b) at least one no-regrets move was identified that's robust across all scenarios; (c) leading indicators are specific enough to actually detect which scenario is unfolding.

## Red Flags

- Building scenarios that are really "optimistic / pessimistic / most likely" variants of a single story rather than structurally distinct futures — the most common failure mode, and one that defeats the entire purpose
- Picking axes that are important-but-predictable or uncertain-but-unimportant instead of the intersection of high-impact and genuinely uncertain
- No leading indicators attached to each scenario — without them, the method can't detect which future is arriving early
- Treating one scenario as "the plan" and the others as decoration, rather than genuinely testing the strategy against all four
- Assigning point probabilities to each scenario and averaging back into a single expected outcome — this collapses the method back into the forecast it was meant to replace

## Verification (Read BEFORE Starting and After Completing)

- [ ] The two axes chosen are both high-impact AND genuinely (not just apparently) uncertain
- [ ] The four resulting scenarios are structurally distinct worlds, not variants of one story
- [ ] Each scenario has an internally consistent narrative and named leading indicators
- [ ] The strategy is checked against all four scenarios, with no-regrets moves separated explicitly from scenario-specific bets
- [ ] No single probability-weighted average is computed across scenarios

## Chaining Protocol

| After completing this skill, check... | Then invoke... |
|---|---|
| Uncertainty resolves enough over time that a calibrated forecast becomes possible | `forecasting-and-measurement` to replace scenario logic with probability logic |
| Need to choose between no-regrets moves and scenario-specific bets | `decision-making-under-uncertainty` for the selection |
| A scenario reveals a failure mode for the current plan | `pre-mortem-and-red-teaming` to develop mitigations |
| Leading indicators start signaling a specific scenario is unfolding | Return to `forecasting-and-measurement` to track the emerging trend |
| The scenarios involve system dynamics (reinforcing/balancing loops) | `systems-thinking` to model the internal structure of each scenario |
