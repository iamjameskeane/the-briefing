"""
Curated transformation pairs demonstrating premium intelligence prose.

These examples show before/after transformations from flat, generic prose
to punchy, Economist/Stratfor-style analysis.

Format:
    Each example is a dict with:
    - before: The flat, satisficing prose
    - after: The transformed, premium prose
    - topic: Topic tag for retrieval (Elections, Conflict, Economic, Diplomatic)
    - tone: Tone tag for retrieval (Somber, Wry, Urgent, Analytical)
    - notes: What makes the transformation work
"""

ECONOMIST_EXAMPLES = [
    {
        "id": "econ_1",
        "topic": "Elections",
        "tone": "Wry",
        "before": """
The election results were significant. The opposition party won more seats than expected. 
This could have implications for government policy. Analysts say this represents a shift 
in voter sentiment. The ruling party will need to reconsider its approach.
""".strip(),
        "after": """
The opposition didn't just win—they swept through districts the ruling party 
considered fireproof. Six cabinet ministers lost their seats. The message 
from voters was not subtle: five years of inflation at 12% buys neither 
loyalty nor patience. The government now faces a choice: pivot on economic 
policy or watch its coalition splinter before summer.
""".strip(),
        "notes": """
Key transformations:
1. Replaced vague "significant" with concrete drama ("swept through fireproof districts")
2. Added specific number (six cabinet ministers, 12% inflation)
3. Changed passive "could have implications" to active prediction
4. Gave the reader a timeline ("before summer")
5. Wry tone through "not subtle" understatement
""",
    },
    {
        "id": "econ_2",
        "topic": "Economic",
        "tone": "Analytical",
        "before": """
The central bank raised interest rates again. This was the fourth increase this year.
The decision was made to combat inflation which remains high. Markets reacted 
negatively to the news. Some economists believe the bank may be raising rates 
too aggressively.
""".strip(),
        "after": """
Four rate rises in eight months—and the central bank shows no signs of blinking. 
At 6.5%, borrowing costs have tripled since January. The strategy is textbook: 
crush demand before inflation expectations become entrenched. But textbooks 
rarely account for an election year. The question is whether the bank's 
independence survives a cabinet that watches its polling numbers crater.
""".strip(),
        "notes": """
Key transformations:
1. Front-loaded the hook with drama ("shows no signs of blinking")
2. Precise numbers (6.5%, tripled, January)
3. Named the tension: textbook economics vs. political reality
4. Ended with a forward-looking question that creates engagement
""",
    },
    {
        "id": "econ_3",
        "topic": "Conflict",
        "tone": "Urgent",
        "before": """
Military tensions increased in the region this week. Both sides conducted 
exercises near the border. Diplomatic talks have stalled. The situation 
remains volatile and could escalate further if no progress is made.
""".strip(),
        "after": """
The buffer zone is shrinking. This week, both armies pushed their exercises 
to within 8km of the demarcation line—half the distance of January's drills. 
Diplomats last spoke on March 3rd and have not scheduled another meeting. 
Without a circuit-breaker, the math is simple: at this rate of compression, 
live fire incidents become probable by April.
""".strip(),
        "notes": """
Key transformations:
1. Opened with a concrete image ("buffer zone is shrinking")
2. Specific distances and dates (8km, January, March 3rd, April)
3. Replaced "could escalate" with probability-based prediction
4. Active voice throughout
5. "Circuit-breaker" terminology adds professional credibility
""",
    },
    {
        "id": "econ_4",
        "topic": "Diplomatic",
        "tone": "Analytical",
        "before": """
The summit between the two leaders was described as productive. They discussed 
trade issues and security cooperation. A joint statement was released 
emphasizing their commitment to dialogue. However, no concrete agreements 
were announced.
""".strip(),
        "after": """
Seventy-two hours of meetings yielded a one-page communiqué and zero 
signatures. The leaders left Davos with their smiles intact and their 
red lines unmoved. Both sides declared the summit "productive"—diplomatic 
code for "we agreed to keep talking." The real tell: neither mentioned 
the tariff dispute in public. What cannot be solved is often simply 
not discussed.
""".strip(),
        "notes": """
Key transformations:
1. Specifics (72 hours, one-page, zero signatures)
2. Decoded diplomatic language ("productive" = "agreed to keep talking")
3. Identified the absence as significant (tariff dispute not mentioned)
4. Closed with an aphorism that crystallizes the insight
""",
    },
    {
        "id": "econ_5",
        "topic": "Economic",
        "tone": "Somber",
        "before": """
The country's debt situation has worsened. The debt-to-GDP ratio has 
increased significantly. International lenders are expressing concern. 
The government is considering austerity measures which could affect 
public services and social programs.
""".strip(),
        "after": """
Debt at 127% of GDP. Interest payments consuming a quarter of tax revenue. 
The IMF delegation arrived Monday; they leave Friday with a reform package 
or a warning. For 4 million pensioners, the calculus is simpler: the 
12% cut being discussed in cabinet meetings translates to choosing 
between heating and medication this winter. Sovereign insolvency is 
an abstraction; hypothermia is not.
""".strip(),
        "notes": """
Key transformations:
1. Led with brutal specifics (127%, quarter of revenue)
2. Concrete timeline (Monday to Friday)
3. Humanized the abstraction (4 million pensioners, heating vs. medication)
4. Final line creates emotional resonance while remaining analytical
""",
    },
]


STRATFOR_EXAMPLES = [
    {
        "id": "strat_1",
        "topic": "Conflict",
        "tone": "Analytical",
        "before": """
Russia has been increasing its military presence near Ukraine. NATO has 
responded with additional deployments. The situation is creating tensions 
in the region. Both sides accuse each other of provocation.
""".strip(),
        "after": """
Geography dictates strategy. The North European Plain—flat, indefensible, 
200 miles wide—is why Moscow sees NATO's eastern expansion as existential 
rather than theoretical. Ukraine is not a prize; it is a buffer. This week's 
troop movements are symptoms, not causes. Until either side finds a way to 
address the underlying geometry of insecurity, expect exercises, not 
de-escalation.
""".strip(),
        "notes": """
Key transformations:
1. Started with geopolitical constraint ("Geography dictates strategy")
2. Explained the structural driver (North European Plain)
3. Reframed the narrative (Ukraine as buffer, not prize)
4. Prediction grounded in structural analysis
""",
    },
    {
        "id": "strat_2",
        "topic": "Diplomatic",
        "tone": "Analytical",
        "before": """
China is investing heavily in African infrastructure. This is part of 
the Belt and Road Initiative. Some analysts see this as expanding 
Chinese influence. African nations benefit from the investment but 
may become dependent on Chinese financing.
""".strip(),
        "after": """
Beijing's African strategy is not charity—it is logistics. Control 
the port at Djibouti, and you control access to the Suez Canal. Finance 
the railway from Mombasa to Nairobi, and you own the trade route to 
East Africa's interior. The loans are secured against exactly what 
China needs: cobalt, copper, and chokepoints. When Western analysts 
call this "debt-trap diplomacy," they miss the point. China is not 
trapping nations—it is purchasing options on the 21st century's 
supply chains.
""".strip(),
        "notes": """
Key transformations:
1. Named the strategic logic ("logistics, not charity")
2. Specific infrastructure (Djibouti port, Mombasa-Nairobi railway)
3. Listed the actual collateral (cobalt, copper, chokepoints)
4. Reframed the "debt trap" narrative with alternative interpretation
""",
    },
    {
        "id": "strat_3",
        "topic": "Economic",
        "tone": "Urgent",
        "before": """
Oil prices have been volatile recently. OPEC+ is considering production 
cuts. The global economy could be affected by higher energy costs. 
Consumers may face higher prices at the pump.
""".strip(),
        "after": """
OPEC+ meets Thursday with one item on the agenda: remind markets who 
controls the tap. The 2 million barrel cut being discussed would push 
Brent above $100—manageable for economies with reserves, catastrophic 
for those without. The unspoken target is Washington, which drained 
half its Strategic Petroleum Reserve last year. The cartel's message 
is blunt: elections have consequences, but so does energy dependence.
""".strip(),
        "notes": """
Key transformations:
1. Specific day and number (Thursday, 2 million barrels, $100)
2. Named the real strategic intent (targeting US SPR depletion)
3. Active language ("remind markets who controls the tap")
4. Ended with geopolitical framing, not consumer prices
""",
    },
]


def get_all_examples() -> list[dict]:
    """Return all curated examples."""
    return ECONOMIST_EXAMPLES + STRATFOR_EXAMPLES
