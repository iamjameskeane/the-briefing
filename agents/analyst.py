"""
Analyst Agent - Pure Reasoning Layer

This agent produces STRUCTURED ANALYTICAL DATA, not prose.
It implements:
- Step-Back Abstraction (geopolitical archetype identification)
- Constraints-of-Thought (Const-o-T) for each actor
- Futures Wheel (1st, 2nd, 3rd order effects)
- Analysis of Competing Hypotheses (ACH)
- PMESII-PT tagging

Changes:
- Works on THEMATIC CLUSTERS, not regions
- Uses FUNCTION CALLING to get context for countries/actors
- Analyst decides which countries/actors need context

The output is JSON that feeds into the Writer Agent.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from config import get_config
from utils import logger, with_retry, log_model_config
from tools import get_tavily_search_tool, execute_tool_call
from .schemas import AnalystInput, AnalystOutput

if TYPE_CHECKING:
    pass


# =============================================================================
# ANALYST PROMPT
# =============================================================================

ANALYST_SYSTEM_PROMPT = """You are a Senior Strategic Analyst with 25 years of experience at the CIA and Stratfor.

Your task is ANALYSIS, not writing. You produce structured reasoning data in JSON format.
Do NOT write prose. Do NOT summarize events. ONLY produce structured analytical frameworks.

## CORE PHILOSOPHY

Geopolitics is driven by CONSTRAINTS, not personalities.
- BAD: "Putin invaded because he is aggressive"
- GOOD: "Moscow lacks natural defensive barriers on the North European Plain, forcing any Russian government to seek strategic depth"

Focus on what actors CANNOT do, not what they choose to do.

## THEMATIC ANALYSIS MODE

You are analyzing a THEMATIC CLUSTER of related events. These events are grouped by their semantic connection, not by geography. A cluster might span multiple regions.

**Your first task**: Use the search_web tool (from tools.py) to gather current context for major countries and actors involved in these events. Search judiciously - only when you need to verify recent developments or actor context.

## REQUIRED ANALYSIS STEPS

### Step 0: GATHER CONTEXT (USE SEARCH_WEB)
- Identify the 2-4 major countries involved in this theme
- Use search_web for each if you need current context (3-8 searches total)
- Identify the 2-4 key actors (leaders, organizations)
- Use search_web for actors if needed (specific queries like "Actor Name role 2026")
- Use this context to inform your analysis

### Step 1: STEP-BACK ABSTRACTION
Before analyzing specifics, identify the dominant geopolitical framework.

CHOOSE THE MOST SPECIFIC ARCHETYPE THAT FITS. Avoid defaulting to "Security Dilemma" or "Balance of Power" for every cluster. Consider:
- Security Dilemma: Actions to increase one's security decrease others' (AVOID overuse)
- Thucydides Trap: Rising power threatens established power
- Resource Curse: Natural resource wealth undermines governance
- Diversionary War: External conflict to distract from domestic problems
- Finlandization: Small state accommodation to great power
- Balance of Power: States align against the strongest (AVOID overuse)
- Heartland Theory: Control of Eurasia = control of world
- Proxy War: Great powers compete through client states
- Containment: Preventing adversary expansion
- Economic Coercion: Using trade/sanctions as primary tool
- Regional Hegemony: Dominant power asserting sphere of influence
- Failed State Contagion: Instability spreading across borders
- Globalization Backlash: Nationalist reaction to economic integration
- Technology Competition: States competing for tech supremacy

Pick the archetype that BEST explains the specific dynamics at play, not just a general catch-all.

### Step 2: CONSTRAINTS-OF-THOUGHT (Const-o-T)
For each major actor (2-4 actors):
1. Identify Strategic Intent (what they want)
2. List Immutable Constraints:
   - Geographic: Physical limits (borders, terrain, chokepoints)
   - Economic: Resource dependencies, trade flows, sanctions
   - Political: Domestic pressures, coalition dynamics, elections
3. Derive Likely Action: What they're FORCED to do given constraints

### Step 3: FUTURES WHEEL (Causal Mapping)
Select the "Driver" event (highest impact, most consequential).
Trace the causal chain:
- 1st Order: Immediate operational consequence (tomorrow)
- 2nd Order: Regional diplomatic/economic reaction (30 days)
- 3rd Order: Long-term structural shift (6 months) - THIS IS THE "SO WHAT?"

The 3rd order effect is what makes analysis valuable. It answers WHY anyone should care.

### Step 4: ANALYSIS OF COMPETING HYPOTHESES (ACH)
- H1 (Consensus): Your main assessment
- H2 (Contrarian): The opposite view - what if you're wrong?
- Contradicting Evidence: Facts that support H2 over H1

This prevents confirmation bias and creates nuance.

### Step 5: PMESII-PT TAGGING
Categorize events by dimension:
- Political: Government actions, policy changes, elections
- Military: Force movements, exercises, weapons systems
- Economic: Trade, sanctions, investments, currency
- Social: Protests, demographics, public opinion
- Infrastructure: Energy, transport, communications
- Information: Media, propaganda, cyber operations

### Step 6: GEOSPATIAL ANCHORING (Mackinder/Spykman)

Tag the region's strategic position:
- **Heartland**: Eurasian core (Russia, Central Asia) - land power dynamics
- **Rimland**: Coastal fringes (W. Europe, Middle East, E. Asia) - maritime trade impact
- **Offshore Balancer**: US, UK, Japan - maintaining equilibrium

Consider: How does geography constrain options? What chokepoints matter?

### Step 7: PRIORITY INTELLIGENCE REQUIREMENTS (PIR)

For critical developments, identify:
- **PIR**: The high-level question (e.g., "Is regime stability threatening energy exports?")
- **EEIs**: Essential Elements of Information - specific observable indicators
  - Physical actions over rhetoric (troop movements > speeches)
  - Quantitative data (shipping volumes, currency flows)
  - Primary sources over secondary (official statements, satellite imagery)

## OUTPUT FORMAT

Respond with valid JSON matching the AnalystOutput schema.
Every field is required. No prose, no explanations outside the JSON structure.
"""


# =============================================================================
# GEOSPATIAL CONSTANTS
# =============================================================================

HEARTLAND_REGIONS = [
    "russia", "central asia", "kazakhstan", "uzbekistan", "turkmenistan",
    "mongolia", "siberia", "caucasus"
]

RIMLAND_REGIONS = [
    "europe", "western europe", "eastern europe", "middle east", "gulf",
    "south asia", "india", "pakistan", "southeast asia", "east asia",
    "china", "japan", "korea", "taiwan"
]

OFFSHORE_BALANCERS = [
    "united states", "usa", "americas", "united kingdom", "uk", "britain",
    "japan", "australia"
]


def get_geospatial_classification(region: str) -> str:
    """Classify a region according to Mackinder/Spykman geopolitics."""
    region_lower = region.lower()
    
    if any(hr in region_lower for hr in HEARTLAND_REGIONS):
        return "HEARTLAND"
    elif any(rr in region_lower for rr in RIMLAND_REGIONS):
        return "RIMLAND"
    elif any(ob in region_lower for ob in OFFSHORE_BALANCERS):
        return "OFFSHORE_BALANCER"
    else:
        return "OTHER"


def _build_analyst_prompt(input_data: AnalystInput) -> str:
    """Build the user prompt for the analyst agent."""

    # Format events for the prompt
    events_summary = []
    for event in input_data.events[:30]:  # Limit to top 30 events
        events_summary.append(
            {
                "id": event.get("id", "unknown"),
                "title": event.get("title", ""),
                "summary": event.get("summary", "")[:300],
                "severity": event.get("severity", 5),
                "category": event.get("category", ""),
                "location": event.get("location_name", ""),
            }
        )

    # Extract countries mentioned for hints
    countries_hint = set()
    for event in input_data.events[:30]:
        loc = event.get("location_name", "")
        if loc:
            countries_hint.add(loc.split(",")[0].strip())

    depth_instruction = ""
    if input_data.is_featured:
        depth_instruction = """
## FEATURED ANALYSIS - MAXIMUM DEPTH REQUIRED

This is the FEATURED ANALYSIS for this week's briefing.
- Use search_web for MORE countries/actors (6-8 searches for featured analysis)
- Analyze 3-4 actors (not just 2)
- Provide more detailed constraints for each actor
- Ensure 3rd order effect is truly structural (changes the balance of power)
- Include more contradicting evidence
"""

    # Use theme label if available, else cluster_id, else region
    identifier = input_data.theme_label or input_data.cluster_id or input_data.region or "Unknown Theme"
    
    regions_info = ""
    if input_data.regions_touched:
        regions_info = f"\n**Regions touched by this theme:** {', '.join(input_data.regions_touched)}"

    prompt = f"""
## THEMATIC CLUSTER: {identifier}
{regions_info}

{depth_instruction}

## EVENTS TO ANALYZE

```json
{json.dumps(events_summary, indent=2)}
```

## YOUR TASK

1. **FIRST**: Use the search_web tool (from tools.py) to gather current context for major countries and actors if needed (3-8 searches total).
   - Countries that might be relevant: {', '.join(list(countries_hint)[:5]) if countries_hint else 'Determine from event content'}
   - Use specific queries like "Country context 2026" or "Actor Name role 2026"
   
2. **THEN**: Produce your analysis using the gathered context.

3. **TRACK SOURCES**: If you used search_web, include the external sources in your output.
   - Extract the key sources you cited from your search results
   - Format: {{"title": "Article Title", "url": "https://...", "publisher": "Source Name"}}
   - Only include sources you actually used for your analysis

Respond with valid JSON matching the AnalystOutput schema.

Required fields:
- region: string (primary region or "THEMATIC" for cross-regional themes)
- geopolitical_archetype: string (one of the frameworks listed)
- archetype_explanation: string
- primary_actors: list of ActorAnalysis objects
- futures_wheel: FuturesWheel object
- competing_hypotheses: CompetingHypothesis object
- pmesii_tags: dict mapping dimension names to lists of event IDs
- confidence: "HIGH" | "MODERATE" | "LOW"
- confidence_rationale: string
- external_sources: list of dicts (if you used search_web, include sources you cited)
  Format: [{{"title": "...", "url": "...", "publisher": "...", "content": "..."}}]
  Include the relevant content/snippet from each source for verification.
  Leave empty [] if no external sources used
"""

    return prompt


@with_retry(max_attempts=3, initial_delay=2.0, max_delay=30.0)
async def run_analyst_agent(
    input_data: AnalystInput,
    client: genai.Client | None = None,
) -> AnalystOutput:
    """
    Run the Analyst Agent to produce structured analytical data.
    
    Uses function calling for context lookup.

    Args:
        input_data: The input containing events and context
        client: Optional Gemini client (created if not provided)

    Returns:
        AnalystOutput with structured reasoning data
    """
    config = get_config()

    if client is None:
        client = genai.Client(api_key=config.gemini_api_key)

    model = config.models.analyst
    identifier = input_data.theme_label or input_data.cluster_id or input_data.region or "Unknown"
    logger.info(f"   🧠 [{identifier}] Running Analyst Agent with {model}...")

    prompt = _build_analyst_prompt(input_data)

    # Use shared Tavily search tool
    tools = [get_tavily_search_tool()]

    # Configure for function calling - only set params if explicitly configured
    gen_config = types.GenerateContentConfig(
        system_instruction=ANALYST_SYSTEM_PROMPT,
        tools=tools,
    )
    
    # Only set temperature if explicitly configured
    if config.analyst_temperature is not None:
        gen_config.temperature = config.analyst_temperature
    
    # Only set max_output_tokens if explicitly configured
    if config.analyst_max_output_tokens is not None:
        gen_config.max_output_tokens = config.analyst_max_output_tokens
    
    # Only set thinking level if explicitly configured
    if config.analyst_thinking_level is not None:
        gen_config.thinking_config = types.ThinkingConfig(
            thinking_level=config.analyst_thinking_level,
    )

    # Log config for debugging
    log_model_config("Analyst-Research", model, gen_config)

    # Start conversation
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    
    # Agentic loop - handle function calls
    max_turns = 5
    for turn in range(max_turns):
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=gen_config,
        )
        
        # Check if model wants to call functions
        if response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts
            function_calls = [p for p in parts if hasattr(p, 'function_call') and p.function_call]
            
            if function_calls:
                # Add model's response to conversation
                contents.append(response.candidates[0].content)
                
                # Execute function calls and add results
                function_responses = []
                for part in function_calls:
                    fc = part.function_call
                    logger.info(f"      📞 Tool call: {fc.name}({fc.args})")
                    result = execute_tool_call(fc.name, dict(fc.args))
                    function_responses.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response={"result": result}
                        )
                    )
                
                # Add function results to conversation
                contents.append(types.Content(role="user", parts=function_responses))
                continue  # Continue loop for next model response
        
        # No more function calls - model should produce final output
        break
    
    # Now get the final JSON output
    # Make a final call requesting structured JSON
    final_config = types.GenerateContentConfig(
        system_instruction=ANALYST_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_json_schema=AnalystOutput.model_json_schema(),
    )
    
    # Only set temperature if explicitly configured
    if config.analyst_temperature is not None:
        final_config.temperature = config.analyst_temperature
    
    # Only set max_output_tokens if explicitly configured
    if config.analyst_max_output_tokens is not None:
        final_config.max_output_tokens = config.analyst_max_output_tokens
    
    # Log config for debugging
    log_model_config("Analyst-Finalize", model, final_config)
    
    # Add instruction to produce final output
    contents.append(types.Content(
        role="user", 
        parts=[types.Part.from_text(
            text="Now produce your final analysis as JSON matching the AnalystOutput schema. Use the context you gathered from the tool calls."
        )]
    ))
    
    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=final_config,
    )

    # Parse the response
    try:
        result = AnalystOutput.model_validate_json(response.text)
        logger.info(
            f"   ✅ [{identifier}] Analyst complete: "
            f"archetype={result.geopolitical_archetype}, "
            f"actors={len(result.primary_actors)}, "
            f"confidence={result.confidence}"
        )
        return result
    except Exception as e:
        logger.error(f"   ❌ [{identifier}] Failed to parse analyst output: {e}")
        logger.error(f"   Raw response: {response.text[:500] if response.text else 'None'}...")
        raise


# =============================================================================
# SYNC WRAPPER
# =============================================================================


def run_analyst_agent_sync(
    input_data: AnalystInput,
    client: genai.Client | None = None,
) -> AnalystOutput:
    """Synchronous wrapper for run_analyst_agent."""
    import asyncio

    return asyncio.run(run_analyst_agent(input_data, client))
