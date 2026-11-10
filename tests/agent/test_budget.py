import pytest

from forecast_agent.agent.budget import BudgetExceeded, QuestionBudget


def test_question_budget_tracks_steps_tokens_and_cost() -> None:
    budget = QuestionBudget(max_steps=2, max_tokens=20, max_cost_usd=0.5)

    budget.consume_step()
    budget.record_usage(input_tokens=8, output_tokens=4, cost_usd=0.2)

    assert budget.steps_used == 1
    assert budget.tokens_used == 12
    assert budget.cost_used_usd == pytest.approx(0.2)
    assert budget.remaining_tokens == 8


def test_question_budget_rejects_work_that_exceeds_any_cap() -> None:
    step_budget = QuestionBudget(max_steps=1, max_tokens=10, max_cost_usd=0.1)
    step_budget.consume_step()

    with pytest.raises(BudgetExceeded, match="step"):
        step_budget.consume_step()

    token_budget = QuestionBudget(max_steps=2, max_tokens=10, max_cost_usd=0.1)
    with pytest.raises(BudgetExceeded, match="token"):
        token_budget.record_usage(input_tokens=8, output_tokens=3, cost_usd=0.01)

    cost_budget = QuestionBudget(max_steps=2, max_tokens=10, max_cost_usd=0.1)
    with pytest.raises(BudgetExceeded, match="cost"):
        cost_budget.record_usage(input_tokens=1, output_tokens=1, cost_usd=0.2)


def test_question_budget_rejects_unknown_cost_instead_of_guessing() -> None:
    budget = QuestionBudget(max_steps=2, max_tokens=10, max_cost_usd=1.0)

    with pytest.raises(BudgetExceeded, match="cost was not reported"):
        budget.record_usage(input_tokens=1, output_tokens=1, cost_usd=None)
