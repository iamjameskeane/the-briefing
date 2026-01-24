"""
Structure Agent - Section Blueprint Layer

This agent plans the detailed structure of each section:
- Beat-by-beat paragraph plans based on story archetype
- Hook and closing drafts
- Event emphasis priorities
- Word allocation per paragraph

The output is a SectionBlueprint that the Writer follows exactly.

Key Frameworks:
- Archetype Templates: CRISIS, TREND, PIVOT, SLEEPER, COMPETITION, CONSTRAINT
- Hook Types: STATEMENT, QUESTION, SURPRISE, CONTRAST, SCENE
- Closing Types: PREDICTION, WARNING, QUESTION, CALLBACK, IMAGE
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from config import get_config
from state import (
    SectionBlueprint,
    SectionDecision,
    ParagraphPlan,
    EventEmphasis,
)
from utils import logger, with_retry, log_model_config
from .schemas import AnalystOutput

if TYPE_CHECKING:
    pass


# =============================================================================
# ARCHETYPE TEMPLATES
# =============================================================================

ARCHETYPE_TEMPLATES = {
    "LEADER": {
        "description": "Featured analysis with full Economist-style treatment (7 beats)",
        "beats": [
            {"purpose": "The Hook - surprising observation or counter-intuitive fact", "word_ratio": 0.10},
            {"purpose": "The News Peg - immediate event that triggered this analysis", "word_ratio": 0.12},
            {"purpose": "The Nub - explicitly state the core tension or puzzle", "word_ratio": 0.12},
            {"purpose": "The Step-Back - historical or structural context", "word_ratio": 0.16},
            {"purpose": "The Argument - core analysis with supporting evidence", "word_ratio": 0.20},
            {"purpose": "The Counter-Argument - grapple with opposing view", "word_ratio": 0.15},
            {"purpose": "The Prescription - forecast or definitive conclusion", "word_ratio": 0.15},
        ],
        "hook_types": ["SURPRISE", "CONTRAST", "SCENE"],
        "closing_types": ["PREDICTION", "WARNING"],
        "pacing": "Build argument. Start counter-intuitive. Layer evidence. Address dissent. End decisively.",
    },
    "CRISIS": {
        "description": "Acute situation requiring immediate attention",
        "beats": [
            {"purpose": "What just happened", "word_ratio": 0.25},
            {"purpose": "Why it matters NOW", "word_ratio": 0.30},
            {"purpose": "Constraints on response", "word_ratio": 0.25},
            {"purpose": "Outlook and triggers to watch", "word_ratio": 0.20},
        ],
        "hook_types": ["STATEMENT", "SCENE"],
        "closing_types": ["PREDICTION", "WARNING"],
        "pacing": "Urgent. Short sentences. Active verbs. Front-load consequences.",
    },
    "TREND": {
        "description": "Pattern emerging or accelerating over time",
        "beats": [
            {"purpose": "The pattern emerging", "word_ratio": 0.25},
            {"purpose": "Evidence this week", "word_ratio": 0.30},
            {"purpose": "Why accelerating now", "word_ratio": 0.25},
            {"purpose": "Structural implication", "word_ratio": 0.20},
        ],
        "hook_types": ["CONTRAST", "QUESTION"],
        "closing_types": ["PREDICTION", "CALLBACK"],
        "pacing": "Build momentum. Layer evidence. Save structural insight for end.",
    },
    "PIVOT": {
        "description": "Actor changing strategy or alignment",
        "beats": [
            {"purpose": "The old position", "word_ratio": 0.20},
            {"purpose": "What changed and why", "word_ratio": 0.35},
            {"purpose": "Constraints forcing the pivot", "word_ratio": 0.25},
            {"purpose": "Who benefits, who loses", "word_ratio": 0.20},
        ],
        "hook_types": ["CONTRAST", "SURPRISE"],
        "closing_types": ["PREDICTION", "QUESTION"],
        "pacing": "Narrative arc. Before/after contrast. Explain the logic.",
    },
    "SLEEPER": {
        "description": "Quiet development with future significance",
        "beats": [
            {"purpose": "Surface calm", "word_ratio": 0.20},
            {"purpose": "The anomaly beneath", "word_ratio": 0.35},
            {"purpose": "Activation conditions", "word_ratio": 0.25},
            {"purpose": "What to watch for", "word_ratio": 0.20},
        ],
        "hook_types": ["QUESTION", "CONTRAST"],
        "closing_types": ["QUESTION", "CALLBACK"],
        "pacing": "Slow build. Create intrigue. Plant seeds.",
    },
    "COMPETITION": {
        "description": "Two+ actors competing for same objective",
        "beats": [
            {"purpose": "The stakes", "word_ratio": 0.20},
            {"purpose": "Competitor A's position and moves", "word_ratio": 0.25},
            {"purpose": "Competitor B's position and moves", "word_ratio": 0.25},
            {"purpose": "Assessment: who has advantage", "word_ratio": 0.30},
        ],
        "hook_types": ["STATEMENT", "SCENE"],
        "closing_types": ["PREDICTION", "QUESTION"],
        "pacing": "Fair presentation. Parallel structure. Clear verdict.",
    },
    "CONSTRAINT": {
        "description": "Structural factor limiting options",
        "beats": [
            {"purpose": "The binding constraint", "word_ratio": 0.25},
            {"purpose": "How it manifested this week", "word_ratio": 0.30},
            {"purpose": "Why it won't change", "word_ratio": 0.25},
            {"purpose": "Implications for strategy", "word_ratio": 0.20},
        ],
        "hook_types": ["STATEMENT", "SURPRISE"],
        "closing_types": ["PREDICTION", "WARNING"],
        "pacing": "Explanatory. Cause and effect. Structural logic.",
    },
}


# =============================================================================
# STRUCTURE INPUT
# =============================================================================

@dataclass
class StructureInput:
    """Input data for the Structure Agent."""
    
    region: str
    analyst_output: AnalystOutput
    section_decision: SectionDecision  # From Architect
    events: list[dict]  # Raw events for emphasis decisions


# =============================================================================
# STRUCTURE PROMPT
# =============================================================================

STRUCTURE_SYSTEM_PROMPT = """
<role>
You are a Structure Planner for premium intelligence writing.
Your job is to create a detailed BLUEPRINT for a single section.
The Writer will follow your plan paragraph-by-paragraph.
</role>

<philosophy>
Structure determines engagement. The same facts, arranged differently,
create vastly different reader experiences. Your job is to find the
arrangement that maximizes insight and readability.
"The difference between good writing and great writing is structure."
</philosophy>

<archetype_templates>
Each story type has a natural structure. Don't fight it—embrace it.

CRISIS:
1. What just happened (25%)
2. Why it matters NOW (30%)
3. Constraints on response (25%)
4. Outlook and triggers (20%)

TREND:
1. The pattern emerging (25%)
2. Evidence this week (30%)
3. Why accelerating (25%)
4. Structural implication (20%)

PIVOT:
1. The old position (20%)
2. What changed and why (35%)
3. Constraints forcing pivot (25%)
4. Who benefits/loses (20%)

SLEEPER:
1. Surface calm (20%)
2. The anomaly beneath (35%)
3. Activation conditions (25%)
4. What to watch (20%)

COMPETITION:
1. The stakes (20%)
2. Competitor A (25%)
3. Competitor B (25%)
4. Assessment (30%)

CONSTRAINT:
1. The binding constraint (25%)
2. This week's manifestation (30%)
3. Why it won't change (25%)
4. Strategic implications (20%)
</archetype_templates>

<hook_types>
The opening line must grab attention:

STATEMENT: Bold claim that demands engagement
  Example: "Moscow has lost control of its buffer states."

QUESTION: Provocative question that creates curiosity
  Example: "What happens when the world's largest army can't resupply?"

SURPRISE: Counter-intuitive fact that makes reader reconsider
  Example: "The biggest threat to Taiwan isn't an invasion—it's an election."

CONTRAST: Juxtaposition that highlights tension
  Example: "Beijing says peace. Its submarines say otherwise."

SCENE: Vivid moment that anchors abstraction
  Example: "At 3 AM Sunday, the last cargo ship left Odessa."
</hook_types>

<closing_types>
The final line must resonate:

PREDICTION: Concrete forecast with timeline
  Example: "Expect a cabinet reshuffle before March."

WARNING: Risk that demands monitoring
  Example: "If the central bank blinks, the cascade begins."

QUESTION: Forward-looking uncertainty
  Example: "The question is not whether Beijing responds, but how."

CALLBACK: Echo of opening that shows change
  Example: "Moscow's buffer is now a liability."

IMAGE: Memorable visual that crystallizes insight
  Example: "The pipeline map now has more red lines than blue."
</closing_types>

<event_emphasis>
Decide which events from the analyst's data deserve:
- HIGH: Central to the story, multiple mentions
- MEDIUM: Supporting evidence, brief mention
- LOW: Background context only
</event_emphasis>

<output_format>
Respond with valid JSON matching the SectionBlueprint schema.
Every field is required. Be specific in your hook and closing drafts.
</output_format>
"""


def _build_structure_prompt(input_data: StructureInput) -> str:
    """Build the user prompt for the Structure agent."""
    
    # Get archetype template
    archetype = input_data.section_decision.archetype
    template = ARCHETYPE_TEMPLATES.get(archetype, ARCHETYPE_TEMPLATES["TREND"])
    
    # Extract key data from analyst output
    analyst = input_data.analyst_output
    
    # Format events
    events_summary = []
    for event in input_data.events[:20]:
        events_summary.append({
            "id": event.get("id", "unknown"),
            "title": event.get("title", "")[:100],
            "severity": event.get("severity", 5),
        })
    
    prompt = f"""
## SECTION TO STRUCTURE

**Region:** {input_data.region}
**Archetype:** {archetype} — {template['description']}
**Treatment:** {input_data.section_decision.treatment}
**Word Target:** {input_data.section_decision.word_target} words

## ARCHETYPE TEMPLATE

**Beats:**
{json.dumps(template['beats'], indent=2)}

**Recommended Hook Types:** {', '.join(template['hook_types'])}
**Recommended Closing Types:** {', '.join(template['closing_types'])}
**Pacing Guidance:** {template['pacing']}

## ANALYST DATA TO WORK WITH

**Geopolitical Archetype:** {analyst.geopolitical_archetype}
**Explanation:** {analyst.archetype_explanation}

**Primary Actors:**
{json.dumps([{
    'actor': a.actor,
    'intent': a.intent,
    'likely_action': a.likely_action
} for a in analyst.primary_actors], indent=2)}

**Futures Wheel:**
- Driver: {analyst.futures_wheel.driver_event}
- 1st Order: {analyst.futures_wheel.first_order}
- 2nd Order: {analyst.futures_wheel.second_order}
- 3rd Order: {analyst.futures_wheel.third_order}

**Competing Hypotheses:**
- Consensus: {analyst.competing_hypotheses.consensus}
- Contrarian: {analyst.competing_hypotheses.contrarian}

**Confidence:** {analyst.confidence}

## EVENTS AVAILABLE

```json
{json.dumps(events_summary, indent=2)}
```

## YOUR TASK

Create a SectionBlueprint with:

1. **Hook** — Choose a hook type and write a draft opening line
2. **Paragraphs** — Plan each paragraph with:
   - Purpose (what it accomplishes)
   - Word target (must sum to section word target)
   - Key facts (specific from analyst data)
   - Beat (from archetype template)
3. **Event Emphasis** — Which events get HIGH/MEDIUM/LOW emphasis
4. **Closing** — Choose a closing type and write a draft final line

## OUTPUT

Respond with valid JSON matching the SectionBlueprint schema.

Required fields:
- region: string
- archetype: string
- word_target: int
- hook_type: "STATEMENT" | "QUESTION" | "SURPRISE" | "CONTRAST" | "SCENE"
- hook_draft: string (the actual hook line)
- paragraphs: list of ParagraphPlan objects
- key_events: list of EventEmphasis objects
- closing_type: "PREDICTION" | "WARNING" | "QUESTION" | "CALLBACK" | "IMAGE"
- closing_draft: string (the actual closing line)
"""
    
    return prompt


# =============================================================================
# STRUCTURE AGENT
# =============================================================================

@with_retry(max_attempts=3, initial_delay=2.0, max_delay=30.0)
async def run_structure_agent(
    input_data: StructureInput,
    client: genai.Client | None = None,
) -> SectionBlueprint:
    """
    Run the Structure Agent to produce a section blueprint.

    Args:
        input_data: Analyst output and section decision
        client: Optional Gemini client (created if not provided)

    Returns:
        SectionBlueprint with paragraph plans
    """
    config = get_config()

    if client is None:
        client = genai.Client(api_key=config.gemini_api_key)

    model = config.models.structure
    logger.info(f"   📝 [{input_data.region}] Running Structure Agent with {model}...")

    prompt = _build_structure_prompt(input_data)

    # Configure for structured JSON output - only set params if explicitly configured
    gen_config = types.GenerateContentConfig(
        system_instruction=STRUCTURE_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_json_schema=SectionBlueprint.model_json_schema(),
    )
    
    # Only set temperature if explicitly configured
    if config.structure_temperature is not None:
        gen_config.temperature = config.structure_temperature
    
    # Only set max_output_tokens if explicitly configured
    if config.structure_max_output_tokens is not None:
        gen_config.max_output_tokens = config.structure_max_output_tokens

    # Log config for debugging
    log_model_config("Structure", model, gen_config)

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=gen_config,
    )

    # Parse the response
    try:
        result = SectionBlueprint.model_validate_json(response.text)
        
        # Validate word targets
        total_words = sum(p.word_target for p in result.paragraphs)
        target = input_data.section_decision.word_target
        
        if abs(total_words - target) > target * 0.15:
            logger.warning(
                f"   ⚠️ [{input_data.region}] Word allocation mismatch: "
                f"planned {total_words}, target {target}"
            )
        
        logger.info(
            f"   ✅ [{input_data.region}] Structure complete: "
            f"hook={result.hook_type}, "
            f"paragraphs={len(result.paragraphs)}, "
            f"closing={result.closing_type}"
        )
        return result
    except Exception as e:
        logger.error(f"   ❌ [{input_data.region}] Failed to parse Structure output: {e}")
        logger.error(f"   Raw response: {response.text[:500]}...")
        raise


# =============================================================================
# FALLBACK BLUEPRINT
# =============================================================================

def create_fallback_blueprint(
    region: str,
    analyst_output: AnalystOutput,
    section_decision: SectionDecision,
    log_warning: bool = True,
) -> SectionBlueprint:
    """
    Create a basic blueprint if the Structure agent fails.
    Uses mechanical mapping instead of LLM judgment.
    """
    if log_warning:
        logger.warning(f"   ⚠️ [{region}] Using fallback blueprint generation")
    
    archetype = section_decision.archetype
    template = ARCHETYPE_TEMPLATES.get(archetype, ARCHETYPE_TEMPLATES["TREND"])
    word_target = section_decision.word_target
    
    # Create paragraphs from beats
    paragraphs = []
    for beat in template["beats"]:
        paragraphs.append(ParagraphPlan(
            purpose=beat["purpose"],
            word_target=int(word_target * beat["word_ratio"]),
            key_facts=[],  # No specific facts in fallback
            beat=beat["purpose"],
        ))
    
    return SectionBlueprint(
        region=region,
        archetype=archetype,
        word_target=word_target,
        hook_type=template["hook_types"][0],
        hook_draft=f"Developments in {region} this week demand attention.",
        paragraphs=paragraphs,
        key_events=[
            EventEmphasis(
                event_id=analyst_output.futures_wheel.driver_event_id,
                emphasis="HIGH",
                role="Driver event"
            )
        ],
        closing_type=template["closing_types"][0],
        closing_draft=f"The coming weeks will prove decisive for {region}.",
    )


# =============================================================================
# BATCH PROCESSING
# =============================================================================

async def run_structure_agents_parallel(
    inputs: list[StructureInput],
    client: genai.Client | None = None,
) -> dict[str, SectionBlueprint]:
    """
    Run Structure agents in parallel for all sections.
    
    Args:
        inputs: List of StructureInput for each section
        client: Shared Gemini client
        
    Returns:
        Dict mapping region to SectionBlueprint
    """
    import asyncio
    
    logger.info(f"   📝 Running {len(inputs)} Structure agents in parallel...")
    
    async def run_with_fallback(input_data: StructureInput) -> tuple[str, SectionBlueprint]:
        try:
            result = await run_structure_agent(input_data, client)
            return input_data.region, result
        except Exception as e:
            logger.error(f"   ❌ [{input_data.region}] Structure failed: {e}")
            fallback = create_fallback_blueprint(
                input_data.region,
                input_data.analyst_output,
                input_data.section_decision,
            )
            return input_data.region, fallback
    
    results = await asyncio.gather(*[run_with_fallback(inp) for inp in inputs])
    return dict(results)


# =============================================================================
# SYNC WRAPPER
# =============================================================================

def run_structure_agent_sync(
    input_data: StructureInput,
    client: genai.Client | None = None,
) -> SectionBlueprint:
    """Synchronous wrapper for run_structure_agent."""
    import asyncio
    return asyncio.run(run_structure_agent(input_data, client))
