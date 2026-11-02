"""Social-post screening with TypeSafe's Jev or Nace's Drex.

Both answer typed questions with calibrated probabilities. Each StockTwits or
Reddit post is asked two: is it about the instrument, and which way does it
lean on the instrument's stock. Code turns the answers into what the Sentiment
Analyst reads: posts that are clearly about something else are dropped, and a
stance count over the rest heads the source's block.

``NACE_API_KEY`` selects Drex (``NACE_DEFAULT_DECISION_MODEL`` pins the model),
even when a TypeSafe key is also set. Otherwise TypeSafe's own SDK variables
apply, ``TYPESAFE_API_KEY``, ``TYPESAFE_DEFAULT_MODEL`` and ``TYPESAFE_BASE_URL``
(OpenRouter serves Jev at ``https://openrouter.ai/api``). Without a key nothing here runs; if any request fails, the source's posts are kept unscreened and the
block says screening was unavailable.
"""

import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from tradingagents import __version__
from tradingagents.agents.context import resolve_instrument_identity

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.typesafe.ai"
_DEFAULT_MODEL = "jev-latest"
_NACE_BASE_URL = "https://console.nace.ai"
_DREX_MODEL = "drex-max-finance"
# Nace counts Drex traffic per integration by these headers; OpenRouter credits the app by its own.
_CLIENT = f"tradingagents/{__version__}"
_DREX_CLIENT = {"X-Nace-Client": _CLIENT, "User-Agent": _CLIENT}
_OPENROUTER = {"HTTP-Referer": "https://nace.ai", "X-OpenRouter-Title": "Drex for tradingagents"}
_RETRY_STATUSES = (429, 529)    # rate limited, overloaded: back off and retry
_TRANSIENT = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)
_ATTEMPTS = 3
_MAX_WAIT = 30.0
_TIMEOUT = 15.0
_WORKERS = 16                   # well inside the documented 1,200 requests per minute

# A post is dropped only on a clear "not about it"; the uncertain middle stays.
_OFF_TOPIC_BELOW = 0.3
# A stance counts only when the answer is not genuinely uncertain.
_STANCE_CONFIDENCE = 0.5

QUESTIONS = {
    "about": {
        "type": "noul",
        "instructions": "Is `post` about `instrument`: the company, its stock, its products or its outlook?",
        "criteria": {
            "true": "`post` discusses `instrument` itself.",
            "false": "`post` names `instrument` only in passing or in a list of tickers, is spam or "
                     "promotion, or is about a different company.",
        },
    },
    "stance": {
        "type": "choice",
        "instructions": "What does the author of `post` expect for the stock price of `instrument`?",
        "criteria": {
            "bullish": "The author expects `instrument`'s stock to rise, or is buying or holding it long.",
            "bearish": "The author expects `instrument`'s stock to fall, or is selling or shorting it.",
            "neutral": "The author gives no view of their own on `instrument`'s stock: a question, "
                       "news without opinion, or someone else's view quoted.",
        },
    },
}


class TypeSafeError(Exception):
    """A System One request that did not produce answers."""


def provider() -> tuple[str, str, str, str]:
    """Who answers: the System One endpoint, key and model, and the name reports use.

    ``NACE_API_KEY`` is the opt-in to Drex and wins over TypeSafe's variables. The
    name follows the model (OpenRouter's ``nace-ai/drex-max`` included), so Drex
    reached through ``TYPESAFE_BASE_URL`` is Drex.
    """
    if key := os.environ.get("NACE_API_KEY"):
        model = os.environ.get("NACE_DEFAULT_DECISION_MODEL") or _DREX_MODEL
        return f"{_NACE_BASE_URL}/v1/systemone", key, model, "Drex"
    # The base TypeSafe's SDKs read, plus /v1/systemone.
    base = (os.environ.get("TYPESAFE_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("TYPESAFE_DEFAULT_MODEL") or _DEFAULT_MODEL
    name = "Drex" if model.rsplit("/", 1)[-1].startswith("drex") else "Jev"
    return f"{base}/v1/systemone", os.environ.get("TYPESAFE_API_KEY", ""), model, name


def system_one(state, questions: dict) -> dict[str, dict]:
    """Ask ``questions`` about ``state``; return the answers keyed by question name.

    Rate-limit and overload responses and dropped connections are retried with
    backoff, honouring ``Retry-After``; any other failure raises
    ``TypeSafeError`` at once.
    """
    url, key, model, name = provider()
    body = {"state": state, "model": model, "questions": questions}
    headers = {"Authorization": f"Bearer {key}"}
    if name == "Drex":
        headers |= _DREX_CLIENT | (_OPENROUTER if url.startswith("https://openrouter.ai/") else {})
    backoff, retry_after = 1.0, None
    for attempt in range(_ATTEMPTS):
        if attempt:
            time.sleep(retry_after if retry_after is not None else backoff * random.uniform(0.8, 1.2))
            backoff *= 2
        try:
            response = requests.post(url, json=body, headers=headers, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            failure, retry_after = type(exc).__name__, None
            if isinstance(exc, _TRANSIENT):
                continue
            break
        if response.status_code == 200:
            return _answers(response, questions, name)
        failure, retry_after = f"HTTP {response.status_code}", _retry_after(response)
        if response.status_code not in _RETRY_STATUSES:
            break
    raise TypeSafeError(failure)


def _retry_after(response) -> float | None:
    try:
        return min(max(0.0, float(response.headers.get("retry-after"))), _MAX_WAIT)
    except (TypeError, ValueError):
        return None


def _answers(response, questions: dict, name: str) -> dict[str, dict]:
    try:
        payload = response.json()
        answers = payload["answers"]
        if all(answers[q]["type"] == spec["type"] for q, spec in questions.items()):
            logger.debug("%s answered with %s", name, payload.get("model"))
            return answers
    except (ValueError, KeyError, TypeError):
        pass
    raise TypeSafeError("malformed response")


def _stance(answer: dict) -> str:
    choice = answer["choice"]
    if choice not in QUESTIONS["stance"]["criteria"] or answer["confidence"] < _STANCE_CONFIDENCE:
        return "unclear"
    return choice


def jev_screen(ticker: str):
    """A post screen for the social fetchers, or None without a Nace or TypeSafe key.

    The screen takes the post texts and returns one keep flag per post and a
    note line for the top of the source's block.
    """
    _, key, _, screener = provider()
    if not key:
        return None
    name = resolve_instrument_identity(ticker).get("company_name")
    instrument = f"{name} ({ticker})" if name else ticker

    def screen(posts: list[str]) -> tuple[list[bool], str]:
        try:
            answers = _ask_each(instrument, posts)
            keep = [a["about"]["noul"] >= _OFF_TOPIC_BELOW for a in answers]
            stances = [_stance(a["stance"]) for a, kept in zip(answers, keep, strict=True) if kept]
        except (KeyError, TypeError):
            return _unscreened(screener, instrument, posts, "malformed response")
        except TypeSafeError as exc:
            return _unscreened(screener, instrument, posts, str(exc))
        counts = ", ".join(f"{stances.count(s)} {s}" for s in ("bullish", "bearish", "neutral", "unclear"))
        return keep, (
            f"Screened by {screener}: {len(stances)} of the {len(posts)} posts fetched are about "
            f"{instrument}; their stance on its stock: {counts}."
        )

    return screen


def _unscreened(screener: str, instrument: str, posts: list[str], failure: str) -> tuple[list[bool], str]:
    logger.warning("%s screening failed for %s: %s", screener, instrument, failure)
    return [True] * len(posts), f"<{screener} screening unavailable ({failure}); posts are unscreened>"


def _ask_each(instrument: str, posts: list[str]) -> list[dict]:
    """One request per post. The first failure raises at once: requests not yet
    sent are cancelled, and those in flight finish in the background unread."""
    answers: list = [None] * len(posts)
    pool = ThreadPoolExecutor(max_workers=_WORKERS)
    try:
        futures = {
            pool.submit(system_one, {"instrument": instrument, "post": post}, QUESTIONS): i
            for i, post in enumerate(posts)
        }
        for future in as_completed(futures):
            answers[futures[future]] = future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return answers
