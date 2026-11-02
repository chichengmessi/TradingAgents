"""The Portfolio Manager's rating from Drex, against the System One request and response shapes."""

from unittest.mock import MagicMock

import pytest

from tradingagents import __version__
from tradingagents.agents import drex_rating as drex, post_screen
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating

BUY = [0.01, 0.02, 0.07, 0.3, 0.6]          # Sell .. Buy


class _Response:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload
        self.headers = {}

    def json(self):
        return self._payload


def _rated(probabilities):
    return _Response(200, {"model": "drex-max-finance", "answers": {"rating": {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probabilities)),
        "probabilities": {str(i): p for i, p in enumerate(probabilities)},
        "confidence": max(probabilities),
    }}})


class _Calls(list):
    def __init__(self):
        super().__init__()
        self.queue = []


@pytest.fixture
def post(monkeypatch):
    """Queue responses on ``.queue``; the list records each call to requests.post."""
    calls = _Calls()

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return calls.queue.pop(0)

    monkeypatch.setattr(post_screen.requests, "post", fake_post)
    monkeypatch.setattr(post_screen.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_test")
    monkeypatch.delenv("NACE_DEFAULT_DECISION_MODEL", raising=False)
    return calls


def _pm_state():
    return {
        "company_of_interest": "NVDA",
        "past_context": "",
        "risk_debate_state": {
            "history": "Aggressive: add now. Conservative: wait. Neutral: the bull case is stronger.",
            "aggressive_history": "", "conservative_history": "", "neutral_history": "",
            "current_aggressive_response": "", "current_conservative_response": "",
            "current_neutral_response": "", "count": 1,
        },
        "investment_plan": "**Recommendation**: Buy",
        "trader_investment_plan": "**Action**: Buy",
    }


def _pm(rating=PortfolioRating.HOLD):
    """A Portfolio Manager whose LLM rates ``rating``, and the prompts it is sent."""
    prompts = []
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: prompts.append(prompt) or PortfolioDecision(
        rating=rating, executive_summary="Summary.", investment_thesis="Thesis.",
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return create_portfolio_manager(llm), prompts


@pytest.mark.unit
def test_without_a_key_nothing_is_sent_and_the_llm_rates(post, monkeypatch):
    monkeypatch.delenv("NACE_API_KEY")
    pm, prompts = _pm(PortfolioRating.OVERWEIGHT)

    result = pm(_pm_state())

    assert post == []
    assert result["final_rating"] == "Overweight"
    assert "has been set" not in prompts[0]


@pytest.mark.unit
def test_asks_drex_one_score_question_over_the_managers_evidence(post):
    post.queue.append(_rated(BUY))

    _pm()[0](_pm_state())

    url, kwargs = post[0]
    assert url == "https://console.nace.ai/v1/systemone"
    assert kwargs["headers"] == {"Authorization": "Bearer nace_sk_test",
                                 "X-Nace-Client": f"tradingagents/{__version__}",
                                 "User-Agent": f"tradingagents/{__version__}"}
    assert kwargs["json"]["model"] == "drex-max-finance"
    assert kwargs["json"]["questions"] == drex.QUESTIONS
    state = kwargs["json"]["state"]
    assert state["research_plan"] == "**Recommendation**: Buy"
    assert state["trader_proposal"] == "**Action**: Buy"
    assert state["risk_debate"].startswith("Aggressive: add now.")
    assert "NVDA" in state["instrument"]


@pytest.mark.unit
def test_the_model_follows_the_sdk_environment(post, monkeypatch):
    monkeypatch.setenv("NACE_DEFAULT_DECISION_MODEL", "drex-max")
    post.queue.append(_rated(BUY))

    drex.drex_rating({})

    assert post[0][1]["json"]["model"] == "drex-max"


@pytest.mark.unit
@pytest.mark.parametrize(("probabilities", "rating"), [
    ([0.9, 0.05, 0.03, 0.01, 0.01], "Sell"),
    ([0.1, 0.5, 0.3, 0.05, 0.05], "Underweight"),
    ([0.05, 0.1, 0.7, 0.1, 0.05], "Hold"),
    (BUY, "Buy"),
])
def test_the_most_likely_level_is_the_rating(post, probabilities, rating):
    post.queue.append(_rated(probabilities))

    assert drex.drex_rating({}) == rating


@pytest.mark.unit
def test_drexs_rating_is_the_decision_and_the_llm_writes_it_up(post):
    post.queue.append(_rated(BUY))
    pm, prompts = _pm(PortfolioRating.HOLD)

    result = pm(_pm_state())

    assert "The rating has been set from this same evidence: **Buy**." in prompts[0]
    assert result["final_rating"] == "Buy"
    assert result["final_trade_decision"].startswith("**Rating**: Buy")


@pytest.mark.unit
@pytest.mark.parametrize("response", [
    _Response(401), _Response(422), _Response(500),
    _Response(200, {"answers": {"rating": {"type": "score", "score": 3.4}}}),
    _Response(200, {"answers": {"rating": {"type": "noul", "noul": 0.9}}}),
    _Response(200, {"answers": {"rating": {"type": "score", "probabilities": dict.fromkeys("01234")}}}),
])
def test_a_failed_request_leaves_the_rating_to_the_llm(post, response, caplog):
    post.queue.append(response)
    pm, prompts = _pm(PortfolioRating.UNDERWEIGHT)

    result = pm(_pm_state())

    assert result["final_rating"] == "Underweight"
    assert "has been set" not in prompts[0]
    assert "Drex rating unavailable" in caplog.text
