---
name: critical-thinking-and-argument-evaluation
description: Evaluates whether a claim or argument actually holds up — fallacy detection, the Toulmin argument model, steelmanning, and evidence-quality tiering. Use when assessing a vendor pitch, a research claim, an AI-generated statement, a code-review argument, or any claim before accepting it as an input to a decision or forecast.
---

# Critical Thinking and Argument Evaluation

## Quick Reference (Layer 0)

**Trigger:** A claim, argument, or evidence needs vetting before it feeds into any other analysis or decision.
**Process in one line:** Decompose into Toulmin structure (find the unstated warrant) → check for named fallacies → tier the actual evidence quality → steelman before responding.
**Token budget:** ~800-1500 tokens for full application; ~300 for evidence-tier check only.

## Overview

Every other analytical skill in this set assumes its inputs (base rates, evidence, claims) are trustworthy. This is the skill that checks that assumption before the input goes anywhere else — it evaluates the *structure and evidentiary weight* of a claim, independent of who made it, how confidently, or how well-produced the source looks. This matters more, not less, in an era of fluent AI-generated text: a claim can be grammatically perfect, structurally plausible, and completely unsupported, and fluency is not evidence.

## Output Convention

State the verdict on the claim first (holds up / doesn't hold up / holds up with caveats), then show which specific link in the argument chain is weak and what evidence tier the claim actually sits at — not just "this seems questionable." Wrap the Toulmin breakdown/fallacy check/evidence-tiering in a collapsible block (`<details><summary>Show working</summary>...</details>`) so only the verdict and its one-line reason are visible by default — expand only if asked.

## When to Use

- Evaluating a vendor/tool pitch, marketing claim, or "best practice" recommendation
- Checking an AI-generated claim, citation, or benchmark result before repeating it
- A code-review or RFC disagreement that's stuck — likely hiding an unnamed fallacy
- Assessing a research finding, news claim, or viral statistic before trusting it
- Responding to an argument — steelman it first so the response addresses the real position

**When NOT to use:** Purely factual lookups with a single authoritative source and no argument being made (a version number, a spec value) — there's no argument to evaluate.

## Step 0: Locate the Claim in the Toulmin Structure

Before judging an argument, decompose it: **Claim** (what's being asserted) → **Grounds** (the evidence offered) → **Warrant** (the unstated assumption connecting grounds to claim) → **Backing** (support for the warrant itself) → **Rebuttal** (conditions under which the claim wouldn't hold). Most weak arguments have a strong claim and grounds but a shaky, unstated warrant — find it before responding.

## Toolkit A — Fallacy Detection

| Fallacy | Pattern | Example |
|---|---|---|
| **Ad hominem** | Attacking the arguer instead of the argument | "They're not even a real engineer, ignore their point" |
| **Strawman** | Refuting a weaker version of the argument than the one actually made | "So you want zero tests at all?" (when the actual claim was "fewer, better-targeted tests") |
| **False dichotomy** | Presenting two options as exhaustive when others exist | "Either we rewrite everything or we do nothing" |
| **Appeal to authority** | Treating a claim as true because an authority said it, without checking their actual evidence | "That's just how the senior engineer wants it done" |
| **Survivorship bias** | Drawing conclusions only from the cases that "survived" a selection process, ignoring the ones that didn't | Studying successful startups only, ignoring the failed ones with the same strategy |
| **Motte-and-bailey** | Defending a modest, easily-defensible claim (the motte) while actually asserting a much stronger one (the bailey) | Retreating to "I just said it's *possible*" after asserting it as near-certain |
| **Appeal to popularity** | Treating a claim as true because many people believe it | "Everyone's using this framework, so it must be the right choice" |

## Toolkit B — Evidence-Quality Tiering

Rank the evidence actually offered, not the confidence with which it's stated:

| Tier (weakest → strongest) | What it is | Caution |
|---|---|---|
| Anecdote | A single personal experience | Vivid, memorable, and the weakest possible evidence |
| Case study | One documented instance, examined in depth | Useful for hypothesis generation, not for general claims |
| Observational study | Correlational data with no controlled comparison | Vulnerable to confounders — see `causal-reasoning-and-root-cause-analysis` |
| Controlled experiment (RCT/A-B test) | Randomized comparison | Strong for the specific population/context tested; may not generalize |
| Replicated meta-analysis | Multiple independent studies, synthesized | Strongest available tier, still not infallible (publication bias, p-hacking) |

**Practical check:** when a claim cites a source, verify what tier that source's own evidence actually sits at — a footnote is not automatically strong evidence; it's only as strong as what it points to.

## Toolkit C — Steelmanning Before Responding

1. State the argument you're evaluating in its strongest possible form — stronger than the arguer actually stated it, if a stronger version exists.
2. Identify what would have to be true for that strongest version to be correct.
3. Only then respond — to the steelman, not to whatever weaker version is easiest to knock down.

This is a discipline against arguing with a strawman of your own unconscious construction, not just other people's strawmen.

## LLM-Specific Failure Modes

| Failure Mode | How This Skill Guards Against It |
|---|---|
| **Fluency-as-evidence bias** | Explicit evidence-tiering forces checking what's behind a well-written claim, not just how convincing it sounds |
| **Sycophantic agreement** | Steelmanning + Toulmin decomposition create a structured reason to disagree with the user when warranted |
| **Confident confabulation** | The requirement to check cited sources' actual evidence tier catches fabricated or misrepresented references |
| **Mode collapse on the first interpretation** | Steelmanning forces generating the strongest *alternative* version of the argument before responding |

## Effectiveness Signal

This skill worked if: (a) a weak warrant was identified that the surface reading would have missed; (b) an evidence tier was downgraded from what the claim's confidence implied; (c) a steelman revealed a stronger version of the opposing position that changed the response.

## Red Flags

- A claim accepted because of who said it (appeal to authority) rather than what evidence they offered
- A rebuttal that addresses a stronger or weaker version of the argument than the one actually made
- A cited source that, when checked, sits at a much weaker evidence tier than the confidence of the claim implies
- An AI-generated or fluently-written claim accepted without checking its actual evidentiary support
- A false dichotomy narrowing the options before alternatives are considered
- Motte-and-bailey — a claim's strong version being asserted, then retreated to its modest version only when challenged
- No steelman attempted before disagreeing with a position

## Verification (Read BEFORE Starting and After Completing)

- [ ] The claim is decomposed into Toulmin's claim/grounds/warrant/backing, and the weak link (usually the warrant) is identified
- [ ] At least one named fallacy check has been run against the argument
- [ ] Any cited evidence has its actual quality tier checked, not just its existence
- [ ] The argument being responded to was steelmanned first
- [ ] AI-generated or third-party claims are treated as claims to verify, not facts to repeat

## Chaining Protocol

| After completing this skill, check... | Then invoke... |
|---|---|
| The claim being evaluated is a causal assertion ("X caused Y") | `causal-reasoning-and-root-cause-analysis` for proper causal testing |
| The claim is a prediction or forecast | `forecasting-and-measurement` to check calibration and base rates |
| The claim survived scrutiny and now requires a decision | `decision-making-under-uncertainty` to act on it |
| The argument involves a systemic/structural claim | `systems-thinking` to model the claimed structure |
| You need to communicate why a claim doesn't hold up to a non-expert | `translating-expert-tradeoffs-for-non-experts` |
