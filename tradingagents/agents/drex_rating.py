"""The Portfolio Manager's rating from Nace's Drex, when ``NACE_API_KEY`` is set.

Drex speaks TypeSafe's System One API (post_screen resolves the endpoint and
model): it answers typed questions with a probability for each answer. The
Portfolio Manager's evidence is asked one ``score`` question over the five
ratings, and the most likely rating is the run's. Without a key nothing here
runs; if the request fails, the Portfolio Manager rates as it does without Drex.
"""

import logging
import os

from tradingagents.agents.post_screen import TypeSafeError, system_one
from tradingagents.agents.rating import RATINGS_5_TIER

logger = logging.getLogger(__name__)

# A score question's levels run lowest first, and its answer is keyed by level.
_SCALE = tuple(reversed(RATINGS_5_TIER))

QUESTIONS = {
    "rating": {
        "type": "score",
        "instructions": (
            "As the portfolio manager for `instrument`, weigh `risk_debate`, `research_plan` and "
            "`trader_proposal`, with the holdings in `portfolio` and the `lessons` from past "
            "decisions, and rate the position. Conflicting stances alone are not a reason to "
            "Hold: the stronger case decides, sized by how decisively it wins. Hold only when the "
            "evidence is still balanced after weighing, or too thin to support a call. A rating "
            "another party gave is evidence, not the decision. Facts outweigh opinions, and a fact "
            "raised in `risk_debate` outweighs a plan written without it."
        ),
        "criteria": [
            "Sell: exit the position or avoid entry",
            "Underweight: reduce exposure, take partial profits",
            "Hold: maintain the current position, no action needed",
            "Overweight: favorable outlook, gradually increase exposure",
            "Buy: strong conviction to enter or add to the position",
        ],
    },
}


def drex_rating(state: dict) -> str | None:
    """The most likely rating for the Portfolio Manager's ``state``, or None.

    None without ``NACE_API_KEY``, or when the request fails, which is logged.
    """
    if not os.environ.get("NACE_API_KEY"):
        return None
    try:
        answer = system_one(state, QUESTIONS)["rating"]
        probabilities = [float(answer["probabilities"][str(level)]) for level in range(len(_SCALE))]
    except (KeyError, TypeError, ValueError):
        failure = "malformed response"
    except TypeSafeError as exc:
        failure = str(exc)
    else:
        logger.debug("Drex rating probabilities, Sell to Buy: %s", probabilities)
        return _SCALE[probabilities.index(max(probabilities))]
    logger.warning("Drex rating unavailable (%s); the Portfolio Manager rates without it", failure)
    return None
