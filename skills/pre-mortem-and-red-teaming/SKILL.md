---
name: pre-mortem-and-red-teaming
description: Imagines a plan has already failed and works backward to find why, before committing — pre-mortem analysis and adversarial red-teaming. Use when a plan/strategy/decision is about to be finalized and needs stress-testing against failure modes before commitment, when the user wants to find blind spots, or asks "what could go wrong," "poke holes in this," or wants an adversarial review of their own reasoning.
---

# Pre-Mortem and Red-Teaming

## Quick Reference (Layer 0)

**Trigger:** A plan is about to be finalized and needs stress-testing; user asks "what could go wrong" or "poke holes in this."
**Process in one line:** Frame as "this has already failed" → generate causes from 5 lenses (adversary/resource/silent/external/blind-spot) → rank by likelihood×impact×catchability → attach mitigations + monitoring.
**Token budget:** ~1500-2500 tokens for full application; ~500 for top-3 failure modes only.

## Overview

`decision-making-under-uncertainty` picks the best option. This skill stress-tests the chosen option *before* commitment, by imagining it has already failed and working backward to find why — a specifically prospective mirror of `causal-reasoning-and-root-cause-analysis`, which works backward from a failure that already happened. The premortem's real value isn't generic risk brainstorming; it's a validated framing trick (imagining the failure as *already having occurred*) that measurably surfaces more, and more specific, causes than asking "what might go wrong" — and it gives cover to say the thing everyone in the room already half-suspects but hasn't voiced.

## Output Convention

Lead with the top-ranked failure modes and the single highest-priority mitigation, then show the full failure list and reasoning. Wrap the full list/reasoning in a collapsible block (`<details><summary>Show working</summary>...</details>`) so only the top failure modes and mitigation are visible by default — expand only if asked.

## When to Use

- A plan or decision is about to be finalized/committed and is a one-way door (hard or costly to reverse)
- The user asks "what could go wrong," "poke holes in this," or wants an adversarial review of their own plan
- There's a risk of groupthink — the plan has had unusually smooth, uncontested agreement so far
- Stakes are high enough to justify the overhead of a structured adversarial pass

**When NOT to use:** routine, cheaply-reversible everyday decisions; a failure that's already happened (that's `causal-reasoning-and-root-cause-analysis`, not this).

## Step 1: Prospective Hindsight Framing

Frame it explicitly as: *"It is [N months/years] from now. This plan has failed completely. Write the history of that failure."* This exact framing — imagining the outcome has already occurred, rather than asking what might go wrong — is the empirically load-bearing part of the technique. Mitchell, Russo & Pennington (1989) found that "prospective hindsight" increased the number of correctly identified reasons for a future outcome by roughly 30% compared to a neutral framing. Don't skip straight to a risk list; do the framing first.

## Step 2: Generate Failure Causes From Multiple Lenses

Rotate through distinct lenses so the exercise doesn't just re-surface the one failure mode everyone already thought of:

| Lens | Question |
|---|---|
| Adversary | How would someone actively trying to defeat this exploit it? |
| Resource failure | What if budget, time, people, or a key dependency simply didn't show up? |
| Silent failure | What fails quietly, undetected until it's too late to fix cheaply? |
| External shock | What outside event (market, regulatory, personnel) breaks an assumption baked into the plan? |
| Internal blind spot | What does everyone already half-suspect but hasn't said aloud? |

The last lens is the one premortems are specifically designed to unlock: Gary Klein's framing gives psychological cover to voice dissent that direct criticism of a live plan tends to suppress (a direct countermeasure to groupthink, Janis 1972).

## Step 3: Rank and Mitigate

Rank causes by **likelihood × impact × catchability** (how early it would need to be caught to still be fixable). For the top few, produce a concrete mitigation now, or a leading indicator/trip-wire to monitor — a named risk with no mitigation or trip-wire attached isn't yet actionable.

**Write-up structure (borrowed from After-Action Review):** For each top-ranked failure mode, structure the output as:
1. **What was expected** — the plan's assumption
2. **What actually happened** (in the imagined failure)
3. **Why the gap** — the mechanism
4. **What to sustain or change** — the mitigation

**Just Culture rule:** Direct imagined failures at systems, information gaps, and structural conditions — not at a person's judgement. Even when a hypothetical failure was triggered by a decision, ask what about the system made that decision likely or easy, not why the person was wrong. This prevents the exercise from collapsing into blame assignment, which shuts down exactly the candor premortems are designed to unlock (Dekker, *Just Culture*, 2007).

## Step 4 (Escalation for High Stakes): Formal Red Team

For genuinely high-stakes plans, assign an explicit adversarial role — a red team structurally separate from the team that built the plan, mirroring military/intelligence practice (devil's advocacy, CIA structured analytic techniques). Separation matters: the same people who built the plan are the least likely to adopt a genuinely adversarial stance toward it.

## Step 5: Set Up Continuous Threat Monitoring (LOSA)

A premortem that stops at commitment time wastes half its value. Aviation's **Line Operations Safety Audit** (LOSA, 1990s–present) evolved past static premortems into real-time threat monitoring during execution — systematically observing normal operations and categorizing threats/errors/undesired states *as they occur*, before they cascade into incidents.

After the premortem identifies failure modes:

1. **Define observable indicators** for each top-ranked failure mode — what would you see in real-time data, logs, behavior, or status updates if this failure mode were beginning to manifest?
2. **Set a monitoring cadence** — daily check, weekly review, or automated alert depending on the failure mode's speed of onset.
3. **Define escalation triggers** — at what signal strength does "monitoring" become "intervening"? Pre-commit to these thresholds rather than trusting yourself to notice in the moment.
4. **Assign monitoring ownership** — a failure mode no one is watching is not being monitored.

The premortem's output is not just "risks accepted" — it's an operational monitoring checklist for the duration of execution.

## LLM-Specific Failure Modes

| Failure Mode | How This Skill Guards Against It |
|---|---|
| **Sycophantic plan-endorsement** | Prospective-hindsight framing gives structured permission to criticize the plan rather than validate it |
| **Generic risk brainstorming** | Five distinct lenses force specific, non-obvious failure modes rather than a list of vague "things that could go wrong" |
| **Surface-level concerns only** | "Internal blind spot" lens specifically targets what hasn't been said aloud — the hardest category for a model that defaults to the obvious |
| **Risk-without-action** | Ranking + mitigation requirement prevents producing a list of worries with no actionable response |

## Effectiveness Signal

This skill worked if: (a) at least one identified failure mode was genuinely non-obvious (not the first thing anyone would think of); (b) each top risk has a concrete mitigation or monitoring indicator; (c) during execution, the LOSA monitoring catches an early warning before it cascades.

## Red Flags

- Skipping the prospective-hindsight framing and going straight to "what could go wrong" (measurably weaker at generating causes)
- Only the obvious external risk surfaces while the half-suspected internal one goes unsaid — that's the exercise failing at its actual job
- A list of named risks with no concrete mitigation or trip-wire attached to any of them
- Red-teaming performed with no attempt to actually adopt an adversarial stance — confirmation bias undermines the exercise
- Using this on a decision that's cheap to reverse — the overhead isn't justified there

## Verification (Read BEFORE Starting and After Completing)

- [ ] Failure was framed prospectively ("this has already failed, why") not just "what could go wrong"
- [ ] Causes drawn from more than one lens (adversarial, resource, silent, external, internal blind spot)
- [ ] At least one identified cause is something the team suspected but hadn't said aloud
- [ ] Top causes ranked by likelihood × impact × catchability
- [ ] Each top cause has a concrete mitigation or leading indicator, not just a named risk
- [ ] For plans with execution duration >1 week, a monitoring cadence and escalation trigger are defined for each top failure mode (LOSA principle)

## Chaining Protocol

| After completing this skill, check... | Then invoke... |
|---|---|
| A failure mode has quantifiable probability and impact | `forecasting-and-measurement` to estimate its likelihood properly |
| A failure mode is structural (a feedback loop, not a single event) | `systems-thinking` to model the loop and find the structural fix |
| The premortem reveals the plan itself was the wrong choice | Return to `decision-making-under-uncertainty` and re-evaluate |
| The top failure mode involves an engineering trade-off | `triz-inventive-problem-solving` to resolve the contradiction rather than just mitigating it |
| Failure modes from other fields' known patterns would help | `cross-domain-analogy` to import failure patterns from analogous domains |
