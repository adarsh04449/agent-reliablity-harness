"""Hand-written rewordings of the flight-booking instruction. No extra LLM call."""

PARAPHRASES = [
    (
        "Please book Alice Chen exactly one one-way ticket SFO to JFK on 15 October 2026. "
        "She wants to leave in the morning and the fare must stay at or below $400."
    ),
    (
        "Reserve exactly one one-way SFO–JFK flight for passenger Alice Chen, date 2026-10-15. "
        "Morning departure preferred. Maximum price: 400 USD."
    ),
    (
        "Alice Chen needs exactly one one-way booking from San Francisco (SFO) to New York (JFK) "
        "on 2026-10-15. Choose a morning flight that costs no more than four hundred dollars."
    ),
    (
        "Get Alice Chen on exactly one one-way morning flight SFO to JFK for 2026-10-15. "
        "Do not book anything over $400."
    ),
]


def all_paraphrases() -> list[str]:
    return list(PARAPHRASES)


def paraphrase_at(index: int) -> str:
    """Cycle through paraphrases. Used later as the paraphrase-condition instruction."""
    return PARAPHRASES[index % len(PARAPHRASES)]
