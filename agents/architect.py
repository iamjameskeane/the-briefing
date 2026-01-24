"""
Architect Agent - Structure Decisions & Synthesize Narrative Arc

The Architect is Phase 2 of the editorial pipeline. It:
1. Takes the Editor's text-based decisions and structures them into DocumentSkeleton JSON
2. Synthesizes the narrative arc using reasoning about the hub mechanism

This separation ensures:
- Editor focuses on kill/publish decisions and research
- Architect owns conceptual architecture (hub mechanism → narrative arc)
- Schema enforcement for valid JSON output
- Reasoning space for narrative synthesis

The Architect has the best view for narrative arc synthesis because it sees:
- The final document structure (what's published vs killed)
- Hub mechanism (if THEMATIC organization)
- Section groupings and transitions
- The complete shape of the briefing
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from config import get_config
from state import DocumentSkeleton
from utils import logger, with_retry, log_model_config
from .editor import validate_all_transitions

if TYPE_CHECKING:
    from .editor import EditorDecisions


# =============================================================================
# ARCHITECT INPUT
# =============================================================================

@dataclass
class ArchitectInput:
    """
    Input data for the Architect Agent.
    
    Takes the Editor's editorial decisions and conversation history
    to generate structured DocumentSkeleton JSON.
    """
    
    editor_decisions: EditorDecisions  # Output from Editor agent
    total_word_budget: int  # Target total words for briefing


# =============================================================================
# ARCHITECT SYSTEM PROMPT
# =============================================================================

ARCHITECT_SYSTEM_PROMPT = """You are the Architect - you structure editorial decisions and synthesize the narrative arc.

The Editor has made kill/publish decisions and identified the organizing principle.
Your job is to:
1. Extract the Editor's decisions and format them into DocumentSkeleton JSON
2. Synthesize the narrative arc (the week's intellectual through-line)

<role>
You are the Architect - you structure editorial decisions and synthesize the narrative arc.
</role>

<constraints>
1. Output ONLY valid JSON matching the DocumentSkeleton schema
2. ALL fields are required - no omissions
3. Use the exact cluster IDs from the Editor's decisions
4. Preserve the Editor's reasoning in rationale fields
5. If the Editor didn't specify a value, use reasonable defaults (e.g., word_target based on treatment)
</constraints>

<extraction_guidance>
The Editor's brief may be structured or freeform. Look for:
- Cluster names and their treatment decisions (FEATURED, FULL, SHORT, QUICK_HIT, KILL)
- Headlines and angles for published clusters
- Kill reasons and PIC scores for killed clusters
- Organizing principle (THEMATIC, REGIONAL, HYBRID)
- Narrative arc
- Section groupings
- Transitions

If the Editor used the suggested format, extraction is straightforward.
If freeform, extract the key decisions and structure them properly.
</extraction_guidance>

<critical_fields>
Every object needs these fields:

FeaturedPlan:
- region (descriptive name - Editor can customize)
- source_cluster_id (REQUIRED: numeric cluster ID like "cluster_25" for analyst matching)
- headline
- angle
- story_archetype
- word_target
- rationale

SectionDecision:
- region (descriptive name - Editor can customize)
- source_cluster_id (REQUIRED: numeric cluster ID like "cluster_12" for analyst matching)
- headline
- treatment
- archetype
- word_target
- position_rationale
- rationale

KilledRegion:
- region (the cluster ID)
- reason (NOTHING_NEW, BELOW_THRESHOLD, REDUNDANT, HOLD_FOR_NEXT_WEEK)
- pic_score (float)
- events_count
- highest_severity
- brief_explanation

QuickHit:
- region
- headline (max 25 words)
- content (1-2 sentences)
- event_id (primary event ID)

SectionGroup:
- title (evocative header)
- sections (list of cluster IDs)
- group_rationale
</critical_fields>

<reasonable_defaults>
If the Editor didn't specify a value, use these defaults:
- word_target: FEATURED=600, FULL=400, SHORT=200, SLEEPER=150, QUICK_HIT=25
- story_archetype: Infer from Editor's description or use TREND as default
- rationale: Use Editor's reasoning, or summarize why this treatment was chosen
- position_rationale: Based on section order (e.g., "Position 2: High impact, follows featured")
- pic_score: If Editor mentioned PIC, use it; otherwise estimate from treatment level
- events_count: Extract from cluster data if available, otherwise use 0
- highest_severity: Extract from cluster data if available, otherwise use 0
- event_id: Use any event ID from the cluster, or generate placeholder if needed
</reasonable_defaults>

<output_format>
Respond with valid JSON only. No markdown code fences, no explanations.
</output_format>
"""


# =============================================================================
# REASONING PHASE PROMPTS (Step 1)
# =============================================================================

REASONING_SYSTEM_PROMPT = """You are the Architect - you structure editorial decisions and synthesize the narrative arc.

The Editor has made kill/publish decisions and identified the organizing principle.
Your job is to think through and document your architectural decisions in structured text form.

<role>
You are the Architect. You extract the Editor's decisions and reason about:
1. Document structure (what to publish, what to kill, in what order)
2. Hub mechanism abstraction (if THEMATIC organization)
3. Narrative arc synthesis (the week's intellectual through-line)
4. Section groupings and transitions
</role>

<task>
Output your reasoning in structured text with clear section headers.
Use this format:

## ORGANIZING PRINCIPLE
[THEMATIC / REGIONAL / HYBRID]

## FEATURED
Cluster: [cluster_id] ([cluster name])
Headline: [punchy headline]
Angle: [unique angle]
Archetype: [CRISIS/TREND/PIVOT/SLEEPER/COMPETITION/CONSTRAINT/LEADER]
Word Target: [number]
Rationale: [why this deserves featured treatment]

## SECTIONS
[For each section to publish, in order:]
1. Cluster: [cluster_id] ([cluster name])
   Headline: [headline]
   Treatment: [FULL/SHORT/COMBINED/SLEEPER/QUICK_HIT]
   Archetype: [archetype]
   Word Target: [number]
   Position Rationale: [why in this position]
   Rationale: [why this treatment]

## KILLED
[For each killed cluster:]
- Cluster: [cluster_id] ([cluster name])
  Reason: [NOTHING_NEW/BELOW_THRESHOLD/REDUNDANT/HOLD_FOR_NEXT_WEEK]
  PIC Score: [score]
  Events: [count]
  Highest Severity: [number]
  Explanation: [brief explanation]

## QUICK HITS
[If any quick hits:]
1. Region: [name]
   Headline: [one-line headline]
   Content: [1-2 sentence summary]
   Event ID: [event_id]

## SECTION GROUPS
[How sections are grouped:]
Group 1: [evocative title]
- Sections: [cluster_id1, cluster_id2, ...]
- Rationale: [why grouped]

## TRANSITIONS
[Between sections:]
[cluster_id1] → [cluster_id2]: [transition text or HARD_BREAK]

## HUB MECHANISM (if THEMATIC)
Hub: [abstract mechanism - NOT actor-specific]
Explanation: [2-3 sentences explaining the mechanism itself]
Manifestations: [list of region names demonstrating this mechanism]
Angle: [MISSING_THE_POINT/UNINTENDED_CONSEQUENCE/etc]

## NARRATIVE ARC
Arc: [Brief provocative thesis, 15-25 words]
Explanation: [Why this arc was chosen]
</task>

<constraints>
1. Output structured text ONLY - no JSON
2. Use clear section headers (##) for easy parsing
3. Preserve all of the Editor's reasoning
4. If the Editor didn't specify a value, use reasonable defaults
5. Extract exact cluster IDs for analyst matching
</constraints>

<reasoning_space>
Take your time to:
- Abstract the hub mechanism (if THEMATIC) - make it universal, not actor-specific
- Synthesize a provocative narrative arc
- Think through section ordering and transitions
- Consider the intellectual through-line
</reasoning_space>
"""


def _build_reasoning_prompt(editor_brief: str, word_budget: int) -> str:
    """Build the prompt for step 1: reasoning phase."""
    
    is_thematic = "THEMATIC" in editor_brief.upper() or "hub" in editor_brief.lower()
    
    return f"""<context>
The Editor has completed editorial review and made the following decisions:

{editor_brief}
</context>

<task>
Think through and document your architectural decisions in structured text form.

Word budget: {word_budget} words total

{"IMPORTANT: This briefing uses THEMATIC organization. You MUST abstract the hub mechanism." if is_thematic else ""}

Use the section header format specified in your system prompt to make your reasoning clear and parseable.

Key tasks:
1. Extract all editorial decisions (featured, sections, killed, quick hits)
2. {"Abstract the hub mechanism to a universal pattern (NOT actor-specific)" if is_thematic else "Organize sections logically"}
3. Synthesize a brief, provocative narrative arc (15-25 words)
4. Plan section groups and transitions
5. Ensure all cluster IDs are preserved for analyst matching

Output your structured reasoning now.
</task>
"""


# =============================================================================
# FORMATTER PHASE PROMPTS (Step 2)
# =============================================================================

FORMATTER_SYSTEM_PROMPT = """You are a JSON formatter for The Briefing's Architect agent.

Your sole job is to take structured reasoning text and convert it to DocumentSkeleton JSON.

<role>
Extract information from structured text and format as valid JSON matching the DocumentSkeleton schema.
</role>

<constraints>
1. Output ONLY valid JSON matching the DocumentSkeleton schema
2. ALL fields are required - no omissions
3. Use exact cluster IDs from the reasoning text
4. Preserve all rationale and explanation text verbatim
5. No markdown code fences, no explanations - JSON only
</constraints>

<extraction_rules>
- Extract cluster IDs like "cluster_21" from lines like "Cluster: cluster_21 (Trump's...)"
- Convert treatment strings to exact enum values (FULL, SHORT, etc.)
- Convert reason strings to exact enum values (NOTHING_NEW, BELOW_THRESHOLD, etc.)
- Parse numeric values for word_target, pic_score, events_count, highest_severity
- Preserve all text in rationale, explanation, headline, angle, content fields
- For transitions, use the exact key format "cluster_id1 → cluster_id2"
</extraction_rules>

<output_format>
Respond with valid JSON only. No markdown, no code fences, no explanations.
Start with { and end with }
</output_format>
"""


def _build_formatter_prompt(reasoning_text: str) -> str:
    """Build the prompt for step 2: formatting phase."""
    
    return f"""<reasoning>
The Architect has completed their reasoning phase and produced this structured output:

{reasoning_text}
</reasoning>

<task>
Convert the above structured reasoning into valid DocumentSkeleton JSON.

Ensure:
- All cluster IDs are extracted correctly (e.g., "cluster_21")
- All enum fields use exact values from the schema
- All required fields are present
- Text fields preserve the original reasoning verbatim
- Transitions use the format "cluster_id1 → cluster_id2" as keys

Output the complete DocumentSkeleton JSON now.
</task>
"""


# =============================================================================
# HUB MECHANISM ABSTRACTION GUIDANCE
# =============================================================================

def _build_hub_mechanism_guidance() -> str:
    """Build guidance for abstracting hub mechanisms from actor-specific themes."""
    return """
## HUB MECHANISM ABSTRACTION (if THEMATIC organization)

If the Editor selected THEMATIC organization, you must ABSTRACT the hub mechanism.

### CRITICAL: Hub ≠ Featured Story

The hub is the MECHANISM (underlying force), not a story about an actor.
The featured story is the biggest MANIFESTATION (Spoke 1 showing the mechanism).

Think:
- Hub = "Gravity" (the force)
- Featured = "Apple falls from tree" (manifestation of gravity)

### THE ABSTRACTION TEST

Ask: "Could this pattern exist without [specific actor/region]?"

❌ **ACTOR-CENTRIC** (too specific):
- "Trump's Foreign Policy Contradictions" → Only applies to Trump
- "Washington's Transactional Diplomacy" → Only applies to US
- "US Military Interventions" → Only applies to US
- "Netanyahu's Gaza Strategy" → Only applies to Israel

✓ **MECHANISM** (abstract, universal):
- "Economic Coercion as Statecraft" → Could apply to any power
- "Unilateral Action Replacing Multilateral Consensus" → Pattern, not actor
- "Military Intervention Bypassing Diplomatic Pressure" → Mechanism, not who
- "Civilian Infrastructure as Battlefield Target" → Tactic, not specific war

### HOW TO ABSTRACT

**1. Look across ALL published spokes** (not just featured)

List all regions you're publishing. What pattern connects them?

**2. Identify the common mechanism**

What underlying force/pattern operates across multiple stories?

**3. Remove the actor**

Transform: "Trump uses tariffs" → "Tariff coercion as policy tool"

**4. Focus on HOW, not WHO**

- WHO: "Trump's contradictory approach"
- HOW: "Coercion and diplomacy as simultaneous tools"

**5. Make it universal**

Could this mechanism apply to other powers/contexts/time periods?

### ABSTRACTION EXAMPLE

**Editor's hub suggestion:** "Trump's Peace Diplomacy Contradictions"

**Your abstraction process:**
<example_thinking>
Featured: Trump threatens tariffs (Greenland), proposes Gaza peace with fees
Other spokes: Venezuela military action, Ukraine pressure for concessions, Iran military buildup

Pattern across spokes:
- All involve unilateral US actions
- All use non-diplomatic tools (military force, economic coercion)
- All bypass traditional multilateral processes
- All demonstrate leverage over consensus

What's the MECHANISM underneath "Trump's policy"?

It's not unique to Trump. The pattern is: when consensus-building is slow or fails,
major powers resort to unilateral coercion (economic or military) to achieve goals.

Remove actor: Not "Trump's approach" but "Coercion as substitute for diplomacy"
Make universal: Could apply to China, Russia, any major power

✅ **Abstracted hub_mechanism:** "Coercion Replacing Consensus in Great Power Relations"
</example_thinking>

### HUB EXPLANATION CONTENT

The `hub_explanation` should explain the MECHANISM in 2-3 sentences, NOT summarize a story.

❌ **BAD** (story summary):
"President Trump's administration is pursuing a highly unconventional foreign policy 
agenda, simultaneously engaging in 'peace diplomacy' while imposing tariffs and 
threatening allies."
→ This describes WHAT Trump did, not the underlying mechanism

✓ **GOOD** (mechanism explanation):
"When diplomatic consensus fails, leverage fills the void. Major powers increasingly 
wield economic tools—tariffs, sanctions, fees—and military threats as primary 
instruments of coercion, bypassing traditional multilateral frameworks and 
accelerating the shift toward transactional international relations."
→ This explains the FORCE/PATTERN operating across multiple actors

### HUB MANIFESTATIONS

List the region names (spokes) that demonstrate this mechanism.

Example:
```
hub_manifestations: [
  "Trump's Peace Diplomacy",
  "US Maduro Capture / Venezuela Oil Sector Opening", 
  "US, Ukraine, Russia Trilateral Talks",
  "Iran Unrest, US Escalation"
]
```

### THINK IT THROUGH

Use your reasoning to:
1. List all published sections
2. Find the pattern connecting 3+ sections
3. Abstract away the actors
4. Test: "Could this mechanism apply elsewhere?"
5. Write mechanism explanation (not story summary)
"""

# =============================================================================
# ARCHITECT PROMPT
# =============================================================================

def _build_architect_prompt(editor_brief: str, word_budget: int) -> str:
    """Build the prompt for the Architect to structure Editor decisions."""
    
    # Check if THEMATIC organization to include hub guidance
    is_thematic = "THEMATIC" in editor_brief.upper() or "hub" in editor_brief.lower()
    
    return f"""<context>
The Editor has completed editorial review and made the following decisions:

{editor_brief}
</context>

<task>
Your job has three parts:

## PART 1: Structure the Editor's decisions into DocumentSkeleton JSON

Ensure ALL required fields are included:
- featured.region (cluster name from Editor)
- featured.source_cluster_id (CRITICAL: original cluster theme label for analyst matching)
- featured.rationale (why this deserves featured treatment)
- featured.word_target (target word count)
- sections[].region (cluster name from Editor)
- sections[].source_cluster_id (CRITICAL: original cluster theme label for analyst matching)
- sections[].word_target (target word count for each)
- sections[].position_rationale (why in this position)
- sections[].rationale (why this treatment was chosen)
- killed[].reason (NOTHING_NEW, BELOW_THRESHOLD, REDUNDANT, or HOLD_FOR_NEXT_WEEK)
- killed[].pic_score (the PIC score that led to kill decision)
- killed[].events_count (number of events in cluster)
- killed[].highest_severity (highest event severity)
- quick_hits[].content (1-2 sentence summary)
- quick_hits[].event_id (any event ID from the cluster)
- section_groups[].title (evocative group header, NOT "group_title")
- section_groups[].sections (list of cluster IDs, NOT "cluster_ids")

CRITICAL: For source_cluster_id, extract the numeric cluster ID from the Editor's brief.
This is used to match published sections back to their analyst data. Examples:
- "Trump's Geopolitical Deals (cluster_25)" → source_cluster_id: "cluster_25"
- "cluster_12" → source_cluster_id: "cluster_12"
- "Iran Crisis (cluster_12)" → source_cluster_id: "cluster_12"

The cluster_id is the canonical key - do NOT use theme labels as source_cluster_id.

Word budget: {word_budget} words total

## PART 2: Abstract Hub Mechanism (if THEMATIC organization)

{_build_hub_mechanism_guidance() if is_thematic else "Not applicable - using REGIONAL organization."}

## PART 3: Synthesize Narrative Arc

After structuring the decisions, synthesize the narrative arc - the week's intellectual through-line.

The narrative arc should be:
- Brief (think 15-25 words, like a headline)
- A **thesis** or **question**, not a summary
- Provocative - something that makes the reader lean in
- Derived from the hub mechanism (if THEMATIC) or overall pattern (if REGIONAL)
- Quotable and memorable

### Style References

❌ **BAD - Summary** (47 words): 
"President Trump's unconventional foreign policy is reshaping global alliances 
and regional flashpoints, creating both diplomatic openings and new tensions 
from the Middle East to Latin America, while major powers grapple with internal 
shifts and persistent instabilities."

✓ **GOOD - Question** (12 words): 
"American pressure creates openings everywhere—but who will seize them?"

✓ **GOOD - Paradox** (10 words): 
"The West fractures while its adversaries watch—and wait."

✓ **GOOD - Tension** (17 words): 
"Can economic coercion replace military force as statecraft? This week's 
evidence says yes."

✓ **GOOD - Provocative statement** (14 words):
"The hegemon threatens everyone, allies included. Who blinks first determines the decade."

### How to Think Through It

Consider:
1. What is the core tension or paradox in the hub mechanism (if THEMATIC)?
2. Looking at the section groups, what question does this week force?
3. What's the intellectual puzzle the reader should grapple with?
4. How would The Economist frame this week in one sentence?
5. What ending creates maximum reader engagement (? vs ! vs —)?

Think through multiple framings. Consider question form, paradox, or provocative 
statement. Then choose the one that best captures the week's meaning.

---

Generate the complete DocumentSkeleton JSON now with:
- If THEMATIC: abstracted hub_mechanism (not actor-specific) and hub_explanation (mechanism explanation)
- narrative_arc (synthesized provocative thesis)
- All other required fields from PART 1
</task>
"""

    return base_prompt


# =============================================================================
# ARCHITECT AGENT - STEP 1: REASONING
# =============================================================================

@with_retry(max_attempts=3, initial_delay=2.0, max_delay=30.0)
async def _run_architect_reasoning(
    input_data: ArchitectInput,
    client: genai.Client,
) -> str:
    """
    Step 1: Reasoning phase - outputs structured text (not JSON).
    
    This allows the model to think freely about hub mechanism abstraction
    and narrative arc synthesis without the constraint of JSON schema.
    
    Args:
        input_data: Editor decisions and word budget
        client: Gemini client
        
    Returns:
        Structured text with architectural decisions
    """
    config = get_config()
    model = config.models.architect
    
    logger.info(f"   📐 Architect Step 1: Reasoning with {model}...")
    
    # Build conversation from Editor's history + reasoning prompt
    messages = input_data.editor_decisions.conversation_history.copy()
    reasoning_prompt = _build_reasoning_prompt(
        input_data.editor_decisions.editorial_brief,
        input_data.total_word_budget
    )
    messages.append(types.Content(role="user", parts=[types.Part(text=reasoning_prompt)]))
    
    # Configure for text output - only set params if explicitly configured
    reasoning_config = types.GenerateContentConfig(
        system_instruction=REASONING_SYSTEM_PROMPT,
    )
    
    # Only set temperature if explicitly configured
    if config.architect_temperature is not None:
        reasoning_config.temperature = config.architect_temperature
    
    # Only set thinking level if explicitly configured
    if config.architect_thinking_level is not None:
        reasoning_config.thinking_config = types.ThinkingConfig(
            thinking_level=config.architect_thinking_level
        )
    
    # Only set max_output_tokens if explicitly configured
    if config.architect_max_output_tokens is not None:
        reasoning_config.max_output_tokens = config.architect_max_output_tokens
    
    # Log config for debugging
    log_model_config("Architect-Reasoning", model, reasoning_config)
    
    response = await client.aio.models.generate_content(
        model=model,
        contents=messages,
        config=reasoning_config,
    )
    
    # Check finish reason
    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
        if finish_reason and "MAX_TOKENS" in str(finish_reason):
            logger.warning(f"   ⚠️ Reasoning phase truncated (MAX_TOKENS) - may be incomplete")
        elif finish_reason and "STOP" not in str(finish_reason) and str(finish_reason) != "1":
            logger.warning(f"   ⚠️ Reasoning finish_reason: {finish_reason}")
    
    reasoning_text = response.text
    logger.info(f"   ✅ Reasoning complete ({len(reasoning_text)} chars)")
    
    return reasoning_text


# =============================================================================
# ARCHITECT AGENT - STEP 2: FORMATTER
# =============================================================================

@with_retry(max_attempts=3, initial_delay=2.0, max_delay=10.0)
async def _run_architect_formatter(
    reasoning_text: str,
    client: genai.Client,
) -> DocumentSkeleton:
    """
    Step 2: Formatter phase - converts structured text to DocumentSkeleton JSON.
    
    This is a pure extraction task with retry support. If formatting fails,
    we can retry just this step without re-running the expensive reasoning phase.
    
    Args:
        reasoning_text: Structured text from step 1
        client: Gemini client
        
    Returns:
        DocumentSkeleton parsed from JSON
    """
    config = get_config()
    model = config.models.architect
    
    logger.info(f"   📐 Architect Step 2: Formatting to JSON with {model}...")
    
    # Build formatter prompt
    formatter_prompt = _build_formatter_prompt(reasoning_text)
    
    # Configure for structured JSON output - only set params if explicitly configured
    formatter_config = types.GenerateContentConfig(
        system_instruction=FORMATTER_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_json_schema=DocumentSkeleton.model_json_schema(),
    )
    
    # Only set temperature if explicitly configured (otherwise use model default)
    if config.architect_temperature is not None:
        formatter_config.temperature = config.architect_temperature
    
    # Only set max_output_tokens if explicitly configured
    if config.architect_max_output_tokens is not None:
        formatter_config.max_output_tokens = config.architect_max_output_tokens
    
    # Only set thinking level if explicitly configured
    if config.architect_formatter_thinking_level is not None:
        formatter_config.thinking_config = types.ThinkingConfig(
            thinking_level=config.architect_formatter_thinking_level
        )
    
    # Log config for debugging
    log_model_config("Architect-Formatter", model, formatter_config)
    
    response = await client.aio.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text=formatter_prompt)])],
        config=formatter_config,
    )
    
    # Check for truncation
    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
        if finish_reason and "MAX_TOKENS" in str(finish_reason):
            raise ValueError(
                f"Formatter response truncated (MAX_TOKENS). "
                f"Consider increasing max_output_tokens or simplifying reasoning output."
            )
        if finish_reason and "STOP" not in str(finish_reason) and str(finish_reason) != "1":
            logger.warning(f"   ⚠️ Formatter finish_reason: {finish_reason}")
    
    # Parse the structured response
    try:
        result = DocumentSkeleton.model_validate_json(response.text)
        logger.info(f"   ✅ Formatter complete: {len(result.sections)} sections")
        return result
    except Exception as e:
        logger.error(f"   ❌ Failed to parse formatter output: {e}")
        logger.error(f"   Raw response: {response.text[:500]}...")
        raise


# =============================================================================
# ARCHITECT AGENT - ORCHESTRATION
# =============================================================================

async def run_architect_agent(
    input_data: ArchitectInput,
    client: genai.Client | None = None,
    output_dir: "Path | None" = None,
) -> tuple[DocumentSkeleton, str]:
    """
    Run the Architect agent in two steps:
    1. Reasoning phase (text output with thinking)
    2. Formatter phase (JSON output, retryable)

    Args:
        input_data: Editor decisions and word budget
        client: Optional Gemini client (created if not provided)
        output_dir: Optional output directory to save reasoning immediately after step 1

    Returns:
        Tuple of (DocumentSkeleton, reasoning_text)
    """
    from pathlib import Path
    
    config = get_config()

    if client is None:
        client = genai.Client(api_key=config.gemini_api_key)

    # Step 1: Reasoning phase (text output, thinking allowed)
    reasoning_text = await _run_architect_reasoning(input_data, client)
    
    # Save reasoning immediately so it's visible even if step 2 fails
    if output_dir:
        reasoning_file = Path(output_dir) / "agent_outputs" / "02a_architect_reasoning.txt"
        reasoning_file.parent.mkdir(parents=True, exist_ok=True)
        reasoning_file.write_text(reasoning_text)
        logger.info(f"   💾 Saved reasoning to {reasoning_file.name}")
    
    # Step 2: Formatter phase (JSON output, retryable independently)
    # The retry decorator on _run_architect_formatter handles retries
    result = await _run_architect_formatter(reasoning_text, client)
    
    # Log decisions
    logger.info(f"   ✅ Architect complete:")
    logger.info(f"      Featured: {result.featured.region}")
    logger.info(f"      Sections: {len(result.sections)} spokes to publish")
    logger.info(f"      Killed: {len(result.killed)} clusters")
    logger.info(f"      Quick Hits: {len(result.quick_hits)}")
    logger.info(f"      Organizing: {result.organizing_principle}")
    
    # Check hub mechanism abstraction
    if result.organizing_principle == "THEMATIC" and result.hub_mechanism:
        logger.info(f"      Hub Mechanism: {result.hub_mechanism}")
        
        # Warn if hub appears actor-centric
        actor_words = ['trump', 'washington', 'biden', 'putin', 'xi', 'netanyahu', 
                      "us '", 'china ', 'russia ', 'israel ', 'iran ']
        mechanism_lower = result.hub_mechanism.lower()
        
        if any(word in mechanism_lower for word in actor_words):
            logger.warning(
                f"      ⚠️ Hub mechanism may be too actor-specific: '{result.hub_mechanism}'"
            )
            logger.warning(f"      💡 Consider abstracting to universal mechanism")
    
    logger.info(f"      Arc: {result.narrative_arc[:80]}...")
    
    # Validate transition quality
    if result.transitions:
        transition_stats = validate_all_transitions(result.transitions)
        logger.info(
            f"      Transitions: {transition_stats['causal']} causal, "
            f"{transition_stats['hard_breaks']} breaks, "
            f"{transition_stats['filler']} filler, "
            f"{transition_stats['weak']} weak"
        )
        
        # Log warnings for filler transitions
        if transition_stats['warnings']:
            for warning in transition_stats['warnings'][:3]:  # Show first 3
                logger.warning(warning)
    
    # Return both skeleton and reasoning text
    return result, reasoning_text


# =============================================================================
# SYNC WRAPPER
# =============================================================================

def run_architect_agent_sync(
    input_data: ArchitectInput,
    client: genai.Client | None = None,
    output_dir: "Path | None" = None,
) -> tuple[DocumentSkeleton, str]:
    """Synchronous wrapper for run_architect_agent."""
    import asyncio
    return asyncio.run(run_architect_agent(input_data, client, output_dir))
