"""
Negative constraints for style transformation.

These lists define what the Stylist should NEVER produce,
helping prevent caricature and maintain professional tone.
"""

# Phrases that should never appear in premium intelligence writing
CLICHE_LIST = [
    # Weasel words
    "remains to be seen",
    "only time will tell",
    "the jury is still out",
    "it is unclear",
    "it remains unclear",
    "sources say",
    "according to experts",
    
    # Overused metaphors
    "perfect storm",
    "at the end of the day",
    "kick the can down the road",
    "tip of the iceberg",
    "paradigm shift",
    "game changer",
    "moving the needle",
    "low-hanging fruit",
    "boots on the ground",
    "skin in the game",
    "double-edged sword",
    "slippery slope",
    
    # Vague qualifiers
    "very",
    "really",
    "extremely",
    "significantly",
    "substantially",
    "quite",
    "rather",
    "somewhat",
    
    # Filler phrases
    "in terms of",
    "with respect to",
    "in the context of",
    "moving forward",
    "going forward",
    "at this point in time",
    "in the current environment",
    
    # Passive hedging
    "could potentially",
    "might possibly",
    "may or may not",
    "it has been suggested",
    "some analysts believe",
    "many observers think",
    
    # Corporate speak
    "leverage",
    "synergy",
    "bandwidth",
    "circle back",
    "touch base",
    "deep dive",
    "unpack",
    "drill down",
]


# What good intelligence prose should NOT look like
NEGATIVE_CONSTRAINTS = {
    "structure": [
        "Don't start every paragraph the same way",
        "Don't use more than 2 sentences that start with 'The'",
        "Don't have all sentences the same length",
        "Don't put the most important point last",
        "Don't bury the lead in qualifications",
    ],
    "tone": [
        "Don't hedge every prediction",
        "Don't use academia-speak",
        "Don't sound like a press release",
        "Don't be gratuitously contrarian",
        "Don't mistake cynicism for insight",
    ],
    "content": [
        "Don't summarize without analysis",
        "Don't explain what happened without why it matters",
        "Don't predict without stating probability",
        "Don't use abstract nouns when concrete examples exist",
        "Don't quote without context",
    ],
    "style": [
        "Don't use passive voice when active works",
        "Don't use 3 words when 1 will do",
        "Don't use jargon without explanation",
        "Don't use clichés from the banned list",
        "Don't sacrifice clarity for cleverness",
    ],
}


# Orwell's rules from "Politics and the English Language"
ORWELL_RULES = [
    "Never use a metaphor, simile, or other figure of speech which you are used to seeing in print.",
    "Never use a long word where a short one will do.",
    "If it is possible to cut a word out, always cut it out.",
    "Never use the passive where you can use the active.",
    "Never use a foreign phrase, a scientific word, or a jargon word if you can think of an everyday English equivalent.",
    "Break any of these rules sooner than say anything outright barbarous.",
]


# Word substitutions for brevity (Orwell Filter)
BREVITY_SUBSTITUTIONS = {
    "facilitate": "help",
    "implement": "do",
    "utilize": "use",
    "methodology": "method",
    "subsequently": "then",
    "approximately": "about",
    "demonstrate": "show",
    "prioritize": "rank",
    "finalize": "finish",
    "originate": "start",
    "terminate": "end",
    "commence": "begin",
    "endeavor": "try",
    "ascertain": "learn",
    "procure": "get",
    "regarding": "about",
    "concerning": "about",
    "sufficient": "enough",
    "numerous": "many",
    "subsequent": "later",
    "prior to": "before",
    "in order to": "to",
    "due to the fact that": "because",
    "in the event that": "if",
    "at the present time": "now",
    "in the near future": "soon",
    "has the ability to": "can",
    "is in a position to": "can",
    "make a decision": "decide",
    "take into consideration": "consider",
    "give consideration to": "consider",
    "make an adjustment": "adjust",
    "perform an analysis": "analyze",
    "conduct an investigation": "investigate",
}


def get_cliche_regex_pattern() -> str:
    """Return a regex pattern that matches any cliché."""
    import re
    escaped = [re.escape(c) for c in CLICHE_LIST]
    return r"\b(" + "|".join(escaped) + r")\b"


def check_for_cliches(text: str) -> list[str]:
    """Return list of clichés found in text."""
    import re
    text_lower = text.lower()
    found = []
    for cliche in CLICHE_LIST:
        if cliche.lower() in text_lower:
            found.append(cliche)
    return found


def apply_brevity_substitutions(text: str) -> str:
    """Apply Orwell-style word substitutions for brevity."""
    import re
    result = text
    for long_word, short_word in BREVITY_SUBSTITUTIONS.items():
        # Case-insensitive replacement preserving original case
        pattern = re.compile(re.escape(long_word), re.IGNORECASE)
        result = pattern.sub(short_word, result)
    return result
