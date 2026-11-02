"""Post screening by Jev or Drex, against the documented System One request and response shapes."""

import pytest
import requests

from tradingagents import __version__
from tradingagents.agents import post_screen as typesafe

QUESTIONS = {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}
ANSWERS = {"is_urgent": {"type": "noul", "noul": 0.95}}


class _Response:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class _Calls(list):
    def __init__(self):
        super().__init__()
        self.queue = []
        self.sleeps = []


@pytest.fixture
def post(monkeypatch):
    """Queue responses on ``.queue``; the list records each call to requests.post."""
    calls = _Calls()
    queue = calls.queue

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(typesafe.requests, "post", fake_post)
    monkeypatch.setattr(typesafe.time, "sleep", calls.sleeps.append)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    monkeypatch.delenv("TYPESAFE_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    return calls


def _ok():
    return _Response(200, {"model": "jev-1.13.0", "answers": ANSWERS,
                           "usage": {"input_tokens": 296, "output_tokens": 20}})


@pytest.mark.unit
def test_sends_the_documented_request_and_returns_the_answers(post):
    post.queue.append(_ok())

    assert typesafe.system_one("Help! My payouts have been failing.", QUESTIONS) == ANSWERS

    url, kwargs = post[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert kwargs["headers"]["Authorization"] == "Bearer ts-test"
    assert kwargs["json"] == {"state": "Help! My payouts have been failing.",
                              "model": "jev-latest", "questions": QUESTIONS}


@pytest.mark.unit
def test_the_model_follows_the_sdk_environment(post, monkeypatch):
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-1.13.0")
    post.queue.append(_ok())

    typesafe.system_one("s", QUESTIONS)

    assert post[0][1]["json"]["model"] == "jev-1.13.0"


@pytest.mark.unit
@pytest.mark.parametrize("base", ["https://openrouter.ai/api", "https://openrouter.ai/api/"])
def test_the_endpoint_follows_the_sdk_environment(post, monkeypatch, base):
    """TypeSafe's SDKs read TYPESAFE_BASE_URL and append /v1/systemone; OpenRouter
    serves Jev at https://openrouter.ai/api with an OpenRouter key (#1416)."""
    monkeypatch.setenv("TYPESAFE_BASE_URL", base)
    post.queue.append(_ok())

    typesafe.system_one("s", QUESTIONS)

    assert post[0][0] == "https://openrouter.ai/api/v1/systemone"


@pytest.mark.unit
def test_a_base_url_without_a_key_does_not_turn_screening_on(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://openrouter.ai/api")
    assert typesafe.jev_screen("AAPL") is None


@pytest.mark.unit
def test_a_nace_key_alone_sends_the_questions_to_drex(post, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    monkeypatch.setenv("NACE_API_KEY", "nace-test")
    post.queue.append(_ok())

    typesafe.system_one("s", QUESTIONS)

    url, kwargs = post[0]
    assert url == "https://console.nace.ai/v1/systemone"
    assert kwargs["headers"]["Authorization"] == "Bearer nace-test"
    assert kwargs["json"]["model"] == "drex-max-finance"


@pytest.mark.unit
def test_the_nace_key_wins_over_typesafes_variables(post, monkeypatch):
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://openrouter.ai/api")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-1.13.0")
    monkeypatch.setenv("NACE_API_KEY", "nace-test")
    monkeypatch.setenv("NACE_DEFAULT_DECISION_MODEL", "drex-max")
    post.queue.append(_ok())

    typesafe.system_one("s", QUESTIONS)

    url, kwargs = post[0]
    assert url == "https://console.nace.ai/v1/systemone"
    assert kwargs["headers"]["Authorization"] == "Bearer nace-test"
    assert kwargs["json"]["model"] == "drex-max"


_DREX_CLIENT = {"X-Nace-Client": f"tradingagents/{__version__}",
                "User-Agent": f"tradingagents/{__version__}"}
_OPENROUTER = {"HTTP-Referer": "https://nace.ai", "X-OpenRouter-Title": "Drex for tradingagents"}


@pytest.mark.unit
@pytest.mark.parametrize(("env", "attribution"), [
    ({}, {}),
    ({"NACE_API_KEY": "nace-test"}, _DREX_CLIENT),
    ({"TYPESAFE_BASE_URL": "https://console.nace.ai", "TYPESAFE_DEFAULT_MODEL": "drex-max-finance"},
     _DREX_CLIENT),
    ({"TYPESAFE_BASE_URL": "https://openrouter.ai/api", "TYPESAFE_DEFAULT_MODEL": "nace-ai/drex-max"},
     _DREX_CLIENT | _OPENROUTER),
])
def test_drex_requests_name_the_client_and_jev_requests_are_unchanged(post, monkeypatch, env,
                                                                      attribution):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    post.queue.append(_ok())

    typesafe.system_one("s", QUESTIONS)

    key = env.get("NACE_API_KEY", "ts-test")
    assert post[0][1]["headers"] == {"Authorization": f"Bearer {key}", **attribution}


@pytest.mark.unit
@pytest.mark.parametrize("transient", [
    _Response(429), _Response(529), requests.ConnectionError(), requests.Timeout(),
    requests.exceptions.ChunkedEncodingError(),
])
def test_rate_limits_overload_and_dropped_connections_are_retried(post, transient):
    post.queue.extend([transient, _ok()])

    assert typesafe.system_one("s", QUESTIONS) == ANSWERS
    assert len(post) == 2


@pytest.mark.unit
def test_a_retry_after_header_sets_the_wait(post):
    post.queue.extend([_Response(429, headers={"retry-after": "7"}), _ok()])

    typesafe.system_one("s", QUESTIONS)

    assert post.sleeps == [7.0]


@pytest.mark.unit
def test_a_long_retry_after_is_capped(post):
    post.queue.extend([_Response(529, headers={"retry-after": "600"}), _ok()])

    typesafe.system_one("s", QUESTIONS)

    assert post.sleeps == [typesafe._MAX_WAIT]


@pytest.mark.unit
def test_retries_are_bounded(post):
    post.queue.extend([_Response(529)] * 3)

    with pytest.raises(typesafe.TypeSafeError, match="HTTP 529"):
        typesafe.system_one("s", QUESTIONS)
    assert len(post) == 3


@pytest.mark.unit
@pytest.mark.parametrize("error", [requests.exceptions.InvalidHeader(), requests.exceptions.TooManyRedirects()])
def test_other_request_errors_are_screening_failures_without_retry(post, error):
    post.queue.append(error)

    with pytest.raises(typesafe.TypeSafeError, match=type(error).__name__):
        typesafe.system_one("s", QUESTIONS)
    assert len(post) == 1


@pytest.mark.unit
@pytest.mark.parametrize("status", [401, 422, 500])
def test_other_failures_raise_without_retry(post, status):
    post.queue.append(_Response(status))

    with pytest.raises(typesafe.TypeSafeError, match=f"HTTP {status}"):
        typesafe.system_one("s", QUESTIONS)
    assert len(post) == 1


@pytest.mark.unit
@pytest.mark.parametrize("payload", [
    None, {"model": "jev"}, {"answers": {"other": {}}},
    {"answers": {"is_urgent": {"type": "choice", "choice": "yes"}}},
])
def test_a_response_without_every_answer_is_an_error(post, payload):
    post.queue.append(_Response(200, payload))

    with pytest.raises(typesafe.TypeSafeError, match="malformed"):
        typesafe.system_one("s", QUESTIONS)



def _post_answers(about: float, stance: str = "bullish", confidence: float = 0.9):
    return _Response(200, {"model": "jev-1.13.0", "answers": {
        "about": {"type": "noul", "noul": about},
        "stance": {"type": "choice", "choice": stance, "confidence": confidence,
                   "probabilities": {stance: 1.0}},
    }})


@pytest.fixture
def jev(post, monkeypatch):
    """Answer each post by its text: ``post`` maps text -> response."""
    answers = {}

    def fake_post(url, **kwargs):
        post.append((url, kwargs))
        return answers[kwargs["json"]["state"]["post"]]

    monkeypatch.setattr(typesafe.requests, "post", fake_post)
    monkeypatch.setattr(typesafe, "resolve_instrument_identity",
                        lambda t: {"company_name": "NVIDIA Corporation"})
    return answers


@pytest.mark.unit
def test_no_key_no_screen(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert typesafe.jev_screen("NVDA") is None


@pytest.mark.unit
def test_each_post_is_its_own_state_under_the_fixed_questions(jev, post):
    jev["NVDA to 200"] = _post_answers(0.9)

    typesafe.jev_screen("NVDA")(["NVDA to 200"])

    body = post[0][1]["json"]
    assert body["state"] == {"instrument": "NVIDIA Corporation (NVDA)", "post": "NVDA to 200"}
    assert body["questions"] == typesafe.QUESTIONS


@pytest.mark.unit
def test_screen_drops_clear_off_topic_posts_and_counts_confident_stances(jev):
    jev.update({
        "long NVDA": _post_answers(0.95, "bullish"),
        "NVDA puts": _post_answers(0.9, "bearish"),
        "maybe NVDA": _post_answers(0.4, "neutral"),       # uncertain relevance: kept
        "NVDA?": _post_answers(0.8, "bullish", 0.3),       # uncertain stance: unclear
        "NVDA!": _post_answers(0.8, "sideways"),           # not an option: unclear
        "$AAPL $MSFT $NVDA pump": _post_answers(0.1, "bullish"),
    })

    keep, note = typesafe.jev_screen("NVDA")(list(jev))

    assert keep == [True, True, True, True, True, False]
    assert note == ("Screened by Jev: 5 of the 6 posts fetched are about NVIDIA Corporation (NVDA); "
                    "their stance on its stock: 1 bullish, 1 bearish, 1 neutral, 2 unclear.")


@pytest.mark.unit
def test_one_failed_request_leaves_every_post_unscreened(jev):
    jev.update({"a": _post_answers(0.1), "b": _Response(401)})

    keep, note = typesafe.jev_screen("NVDA")(["a", "b"])

    assert keep == [True, True]
    assert note == "<Jev screening unavailable (HTTP 401); posts are unscreened>"


@pytest.mark.unit
def test_an_answer_missing_its_fields_reads_as_malformed(jev):
    jev["a"] = _Response(200, {"answers": {"about": {"type": "noul"},
                                           "stance": {"type": "choice"}}})

    keep, note = typesafe.jev_screen("NVDA")(["a"])

    assert keep == [True]
    assert note == "<Jev screening unavailable (malformed response); posts are unscreened>"


@pytest.mark.unit
@pytest.mark.parametrize("env", [
    {"NACE_API_KEY": "nace-test"},
    {"TYPESAFE_BASE_URL": "https://console.nace.ai", "TYPESAFE_DEFAULT_MODEL": "drex-max-finance"},
    {"TYPESAFE_BASE_URL": "https://openrouter.ai/api", "TYPESAFE_DEFAULT_MODEL": "nace-ai/drex-max"},
])
def test_the_screen_names_drex_when_drex_answers(jev, monkeypatch, env):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    if "NACE_API_KEY" not in env:
        monkeypatch.setenv("TYPESAFE_API_KEY", "nace-test")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    jev.update({"long NVDA": _post_answers(0.95), "b": _Response(401)})

    _, note = typesafe.jev_screen("NVDA")(["long NVDA"])
    _, failed = typesafe.jev_screen("NVDA")(["b"])

    assert note.startswith("Screened by Drex: 1 of the 1 posts fetched")
    assert failed == "<Drex screening unavailable (HTTP 401); posts are unscreened>"


@pytest.mark.unit
def test_the_first_failure_cancels_the_requests_not_yet_sent(jev, post, monkeypatch):
    monkeypatch.setattr(typesafe, "_WORKERS", 1)
    jev.update({"a": _Response(401), **{f"p{i}": _post_answers(0.9) for i in range(20)}})

    typesafe.jev_screen("NVDA")(list(jev))

    assert len(post) < 21


@pytest.mark.unit
def test_the_sentiment_analyst_hands_the_screen_to_both_social_fetchers(monkeypatch):
    from langchain_core.messages import AIMessage

    from tradingagents.agents.analysts import sentiment_analyst

    screen = object()
    seen = []
    monkeypatch.setattr(sentiment_analyst, "jev_screen", lambda ticker: screen)
    monkeypatch.setattr(sentiment_analyst.get_news, "func", lambda *a: "news")
    for name in ("fetch_stocktwits_messages", "fetch_reddit_posts"):
        monkeypatch.setattr(sentiment_analyst, name, lambda *a, screen=None, **k: seen.append(screen) or "")

    class _LLM:
        def with_structured_output(self, *a, **k):
            raise NotImplementedError

        def invoke(self, messages):
            return AIMessage(content="report")

    node = sentiment_analyst.create_sentiment_analyst(_LLM())
    node({"company_of_interest": "NVDA", "trade_date": "2026-01-09", "messages": []})

    assert seen == [screen, screen]


@pytest.mark.unit
@pytest.mark.parametrize(("nace_key", "screener"), [(None, "Jev"), ("nace-test", "Drex")])
def test_the_sentiment_analysts_prompt_names_who_screened(monkeypatch, nace_key, screener):
    from langchain_core.messages import AIMessage

    from tradingagents.agents.analysts import sentiment_analyst

    if nace_key:
        monkeypatch.setenv("NACE_API_KEY", nace_key)
    monkeypatch.setattr(sentiment_analyst, "jev_screen", lambda ticker: None)
    monkeypatch.setattr(sentiment_analyst.get_news, "func", lambda *a: "news")
    for name in ("fetch_stocktwits_messages", "fetch_reddit_posts"):
        monkeypatch.setattr(sentiment_analyst, name, lambda *a, **k: "")
    prompts = []

    class _LLM:
        def with_structured_output(self, *a, **k):
            raise NotImplementedError

        def invoke(self, messages):
            prompts.append(messages[0].content)
            return AIMessage(content="report")

    sentiment_analyst.create_sentiment_analyst(_LLM())(
        {"company_of_interest": "NVDA", "trade_date": "2026-01-09", "messages": []})

    assert f'A block headed "Screened by {screener}" has had' in prompts[0]


@pytest.mark.unit
def test_a_failure_does_not_wait_for_requests_still_in_flight(jev, monkeypatch):
    import threading
    import time

    release = threading.Event()

    class _Slow:
        status_code = 200
        headers = {}

        def json(self):
            release.wait(5)
            return _post_answers(0.9).json()

    jev.update({"slow": _Slow(), "bad": _Response(401)})
    started = time.monotonic()
    keep, note = typesafe.jev_screen("NVDA")(["slow", "bad"])
    elapsed = time.monotonic() - started
    release.set()

    assert keep == [True, True] and "unavailable" in note
    assert elapsed < 2
