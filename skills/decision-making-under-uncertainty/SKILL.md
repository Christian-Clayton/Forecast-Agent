---
name: decision-making-under-uncertainty
description: Applies decision theory to choose the best action when outcomes are uncertain — expected value, decision trees, opportunity cost, real options, and reversible-vs-irreversible framing. Use when the user must choose between options with uncertain payoffs (investments, offers, features, trades, life decisions), is stuck in analysis paralysis, or needs to decide how much decision-making effort a choice actually deserves.
---

# Decision-Making Under Uncertainty

## Quick Reference (Layer 0)

**Trigger:** Must choose between options with uncertain payoffs, stuck in analysis paralysis, or need to calibrate how much decision effort a choice deserves.
**Process in one line:** Gate on reversibility (two-way door = decide fast) → if one-way door: compute EV per option → check opportunity cost + sunk-cost trap → set a tripwire for re-decision.
**Token budget:** ~1500-3000 tokens for full decision tree; ~300 for reversibility gate + quick heuristic.

## Overview

Forecasting answers "how likely is X". This skill answers "given that, what should I *do*" — it's the action layer that sits on top of any probability estimate. A perfectly calibrated forecast is worthless if it isn't converted into a decision correctly: weighted by payoffs, checked against opportunity cost, and matched to how reversible the choice actually is. Decision theory is one of the oldest formalized disciplines here (von Neumann-Morgenstern utility theory, 1944) precisely because "probability times payoff" undersells how easy it is to get this wrong — sunk costs, framing effects, and mismatched effort-to-stakes ratios distort decisions even when the underlying probabilities are correct.

## Output Convention

Same actuarial shape as `forecasting-and-measurement`: **state the recommended action and its expected value first**, then show the tree/calculation, then caveats. Every option compared needs a numeric expected value or an explicit reason it can't be quantified (e.g., "no basis to price this branch — judgment call, flagged"). Wrap the tree/calculation and caveats in a collapsible block (`<details><summary>Show working</summary>...</details>`) so only the recommended action and its expected value are visible by default — expand only if asked.

## When to Use

- Choosing between options with uncertain, quantifiable-ish payoffs (job offers, investments, trades, features, contracts)
- Stuck between "keep researching" and "just decide" (analysis paralysis)
- Deciding how much process a decision deserves (a five-minute call vs. a formal analysis)
- Evaluating whether to wait for more information or commit now
- Catching your own sunk-cost reasoning before it drives a decision

**When NOT to use:** Decisions with no meaningful uncertainty (the right answer is already obvious/deterministic), or purely values-based choices with no payoff structure to model (who to marry, what to believe).

## Step 0: Gate the Decision Before Modeling It

**Q1 — Is this decision reversible?** (Bezos's one-way door / two-way door)
- **Two-way door** (cheap/fast to undo): decide fast, with a low-cost heuristic. Spending a full decision-tree analysis here is itself a decision error — it wastes the one resource (time) that this class of decision doesn't require you to spend.
- **One-way door** (expensive or impossible to undo): deserves the full toolkit below — expected value, decision tree, opportunity cost, real-options check.

**Q2 — Is the payoff structure actually quantifiable?** If you can attach even rough numbers (cost, probability, benefit) to each branch, use Toolkit A. If the decision is genuinely values-based with no payoff structure (not "I don't want to do the math" but "there is no math"), this skill doesn't apply — state that explicitly rather than forcing a fake number onto it.

## Toolkit A — Quantified Decisions

| Technique | What it does | Source |
|---|---|---|
| **Expected value (EV)** | Multiply each outcome's probability by its payoff, sum across outcomes — the baseline calculation before anything fancier | Decision analysis (Howard) |
| **Decision tree** | Branch on each uncertain event, prune dominated branches, roll back from the leaves to find the best current choice | Raiffa; Howard |
| **Opportunity cost** | The value of a choice is what you give up by not taking the next-best alternative, not just the raw payoff of the choice itself | Economics |
| **Marginal analysis** | Only the next unit's cost/benefit matters for the decision at hand — the sunk average is irrelevant | Economics |
| **Minimax regret** | When the downside of one branch is unacceptable regardless of its probability, choose to minimize the worst-case regret rather than maximize EV | Decision theory (Savage) |
| **Real options** | Model the value of *waiting* for more information as its own asset with a price — explains when "not deciding yet" is the mathematically correct move, not indecision | Amram & Kulatilaka |
| **Expected utility (risk aversion)** | Convert payoffs through a utility function before averaging, so a loss counts differently than an equivalent-sized gain when the stakes are large relative to your resources | von Neumann & Morgenstern |
| **Utility elicitation (for non-monetary outcomes)** | For decisions where outcomes aren't naturally in objective units (career, relationship, values-laden calls), use a structured anchoring method: anchor 0 = worst realistic outcome, 1 = best realistic outcome, then locate each intermediate outcome on that scale using a standard gamble or time trade-off — don't let the decision-maker state a payoff intuitively without a structured method behind it | Clinical decision analysis (Torrance, 1986; actively researched 2025 for elicitation method comparison) |
| **Multi-criteria / weighted scoring** | When a decision has several incommensurable objectives (cost, time, risk, quality), assign explicit weights and score each option against each criterion rather than eyeballing a holistic impression | Keeney & Raiffa |

## Toolkit B — Debiasing the Decision Process

| Bias/trap | How it distorts the decision | Corrective |
|---|---|---|
| **Sunk-cost fallacy** | Past (unrecoverable) costs pull toward continuing a bad bet | Recompute EV using only *future* costs and benefits from this point forward |
| **Framing effects** | The same option chosen/rejected depending on whether it's framed as a gain or a loss | Restate the decision in both frames before choosing |
| **Loss aversion overweighting** | Losses loom larger than equivalent gains, distorting the risk premium demanded | Use expected utility explicitly rather than a gut "feels too risky" call |
| **Overconfidence in the estimate feeding the tree** | A decision tree is only as good as its input probabilities | Pull probabilities from `forecasting-and-measurement`'s calibrated-estimate process, not a guess |
| **Analysis-paralysis on a two-way door** | Spending one-way-door-level effort on a reversible decision | Apply the Step 0 gate before starting any analysis |
| **Ignoring the option value of waiting** | Treating "decide now" as the only choice, when "decide later, after cheap information" is itself a branch | Explicitly price a "wait" branch using real-options logic |
| **Resulting (process-outcome confusion)** | Judging a decision's quality by its outcome rather than its process — a bad-luck good-process decision gets wrongly updated away from, a good-luck bad-process decision gets wrongly reinforced | Score decisions on a 2×2: good/bad process × good/bad outcome. Only update your process in response to *process* problems, not outcome noise. Track this ratio over time as a meta-calibration metric (from poker/decision science: Annie Duke, 2018; Tetlock) |

## The WRAP Process (Practical Checklist)

For decisions too fuzzy for a clean tree, use Heath & Heath's WRAP steps to structure the judgment call:
1. **Widen your options** — the default "whether or not" framing hides better alternatives; force at least one more option onto the table.
2. **Reality-test assumptions** — actively seek disconfirming evidence for your current favorite (ties to `forecasting-and-measurement`'s ACH technique).
3. **Attain distance before deciding** — ask what you'd advise a friend in the same spot, or how this looks in 10 minutes/months/years.
4. **Prepare to be wrong** — set a tripwire (a specific, pre-committed signal) that triggers a re-decision, rather than trusting yourself to notice in the moment.

## LLM-Specific Failure Modes

| Failure Mode | How This Skill Guards Against It |
|---|---|
| **False precision** | Reversibility gate prevents over-analyzing trivial choices; explicit "no basis to quantify" flag prevents forcing fake numbers |
| **Sunk-cost honoring** | Explicit marginal-analysis step forces recomputing from current state forward, not from project start |
| **Missing the "do nothing" branch** | Real-options check ensures "wait for more information" is always priced as an explicit branch |
| **Sycophantic option-endorsement** | EV calculation provides a structured reason to recommend against the user's stated preference when the math disagrees |

## Effectiveness Signal

This skill worked if: (a) a two-way-door decision was handled in <5 minutes rather than over-analyzed; (b) a one-way-door decision has an explicit EV comparison that would change the recommendation if inputs changed; (c) a tripwire exists that will trigger re-evaluation rather than silent commitment.

## Red Flags

- A decision analyzed with full decision-tree rigor that was actually a cheap, reversible two-way door
- An expected-value calculation that includes sunk costs
- A single-branch analysis with no "do nothing / wait" option priced
- Probabilities plugged into a tree with no stated source (should come from a calibrated forecast, not a guess)
- A "gut feel" veto overriding a computed EV with no stated reason
- Treating a values-based decision as if it had a quantifiable payoff structure
- No tripwire or reconsideration point set for a decision made under real uncertainty

## Verification (Read BEFORE Starting and After Completing)

- [ ] Decision classified as reversible (two-way door) or not, before choosing how much process to apply
- [ ] Every quantified option has an explicit EV or is flagged as non-quantifiable judgment
- [ ] Sunk costs excluded from the forward-looking calculation
- [ ] A "wait for more information" branch was considered and explicitly priced or dismissed with a reason
- [ ] Opportunity cost (the next-best alternative) is stated, not just the chosen option's raw payoff
- [ ] If probabilities came from a forecast, its calibration/source is stated
- [ ] A tripwire or re-decision point exists for decisions made under real, unresolved uncertainty
- [ ] After resolution, the decision is scored on process quality (good/bad) independently of outcome quality — only process failures trigger a method change

## Chaining Protocol

| After completing this skill, check... | Then invoke... |
|---|---|
| The chosen option is a one-way door with high stakes | `pre-mortem-and-red-teaming` to stress-test before committing |
| The input probabilities are guesses rather than calibrated estimates | `forecasting-and-measurement` to improve the probability inputs |
| The decision involves deep/structural uncertainty with no track record | `scenario-planning` instead of single-point EV |
| A past decision failed and you need to understand why | `causal-reasoning-and-root-cause-analysis` for the mechanism, then score process vs. outcome |
| The decision requires quick sizing of an unknown quantity | `fermi-estimation` for a fast input number |
