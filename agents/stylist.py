"""
Stylist Agent - Voice Transformation Layer

This agent transforms accurate-but-flat prose into punchy, memorable writing:
- Applies style transformation (Economist/Stratfor voice)
- Varies sentence structure
- Adds memorable lines
- Strengthens openings and closings
- Applies Orwell Filter
- Preserves Sacred Elements exactly

The output is polished prose ready for the Critic.

Key Frameworks:
- Few-shot examples from curated bank
- Orwell's Rules for clear writing
- Sacred Elements enforcement
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from config import get_config
from examples import get_style_examples
from config import get_config
from examples.negative_constraints import (
    check_for_cliches,
    apply_brevity_substitutions,
    CLICHE_LIST,
)
from state import SacredElements, SectionBlueprint
from utils import logger, with_retry, log_model_config
from .writer import strip_metadata_labels  # Reuse Writer's label stripper

if TYPE_CHECKING:
    pass


# =============================================================================
# STYLIST INPUT
# =============================================================================

@dataclass
class StylistInput:
    """Input data for the Stylist Agent."""
    
    region: str
    writer_draft: str  # The draft to transform
    blueprint: SectionBlueprint  # Original structure plan
    sacred_elements: SacredElements  # Facts that cannot be altered
    tone: str = "Analytical"  # Target tone
    voice_mode: str = "ECONOMIST_STRATFOR"  # Default and only voice mode
    # Revision context (for retries)
    previous_styled: Optional[str] = None  # Previous styling attempt
    critic_feedback: Optional[str] = None  # What the Critic flagged


# =============================================================================
# VOICE MODE CONFIGURATIONS
# =============================================================================

VOICE_MODES = {
    "ECONOMIST_STRATFOR": {
        "description": "The Economist wit + Stratfor constraint-focus",
        "characteristics": [
            "Wry, understated authority",
            "Depersonalized determinism (states, not leaders)",
            "Precise, dense prose",
            "Constraint-focused analysis",
        ],
        "system_modifier": "",  # Default mode, no modifier needed
    },
}


def get_voice_modifier(voice_mode: str) -> str:
    """Get the system prompt modifier for a voice mode."""
    mode = VOICE_MODES.get(voice_mode, VOICE_MODES["ECONOMIST_STRATFOR"])
    return mode.get("system_modifier", "")


# =============================================================================
# STYLIST PROMPT
# =============================================================================

STYLIST_SYSTEM_PROMPT = """
<role>
You are a Master Stylist for premium intelligence writing.
Your job is VOICE TRANSFORMATION. You take accurate, well-structured prose
and make it sing. You do not change facts—you change how they land.
</role>

<philosophy>
In intelligence writing, SPECIFICITY IS MEMORABILITY.

A vague metaphor is forgettable. A surprising fact is unforgettable.
Your job is not to make prose "literary"—it's to make DATA land with impact.

BAD (poetic but empty): "Demographics are destiny, and Beijing's destiny is shrinking."
GOOD (specific and punchy): "A 17% birth rate collapse has forced Beijing to choose silicon over citizens."

The first is forgettable. The second is quotable.

The constraint phrase IS the hook. The number IS the drama. 
Don't replace data with poetry. Make data poetic.
</philosophy>

<voice_characteristics>
THE ECONOMIST/STRATFOR VOICE:
- Confident without arrogance
- Precise without pedantry
- Wry without snark
- Dense without obscure

Core traits:
- Authoritative but not arrogant
- Concise but not superficial
- Analytical, not emotional
- Uses active voice EXCLUSIVELY ("Tehran escalated" not "Escalation occurred")
- Makes judgments, doesn't hedge everything
- Acknowledges uncertainty explicitly when present
- Uses metonyms for state actors (Washington not Biden, Beijing not Xi)
- Emphasizes CONSTRAINTS over personalities (states are driven by interests)
- Pungent language: Strong verbs and nouns, minimal adjectives and adverbs
</voice_characteristics>

<transformations>
What you transform:

1. SENTENCE VARIETY
   - Vary sentence length: Mix short (6-10 words) with longer (15-25 words)
   - Vary sentence openings - use these tactics:
     * Start with actor: "Washington deployed forces."
     * Start with time/context: "This week, tensions escalated."
     * Start with dependent clause: "Constrained by sanctions, Tehran..."
     * Start with result: "Tensions spiked when..."
     * Start with contrast: "Yet Moscow held firm."
   - AVOID: More than 2 consecutive sentences starting with "The..." or "Washington..." or "This..."
   - FIX REPETITION: If the Writer used the same starter 3+ times, rewrite those sentences

2. WORD CHOICE
   - Replace weak verbs with strong ones
   - Cut unnecessary qualifiers
   - Use concrete over abstract

3. OPENING PUNCH
   - Make the first line irresistible through SPECIFICITY
   - The hook MUST contain: a constraint, a number, or a prediction
   - Never trade specificity for abstraction
   - The data IS the drama—don't replace it with metaphor

4. CLOSING RESONANCE
   - Final line should stick
   - Echo, predict, or provoke

5. THE QUOTABLE LINE
   - One line per section that crystallizes the insight
   - Something a reader would share or remember
</transformations>

<sacred_elements>
THESE MUST NOT BE CHANGED, PARAPHRASED, OR EMBELLISHED:
- Proper nouns (names exactly as given)
- Statistics (numbers exactly as given)
- Dates (exactly as given)
- Quotes (word-for-word as given)

If you change a Sacred Element, you have failed.
</sacred_elements>

<analytical_content_preservation>
CRITICAL: You are a STYLIST, not an EDITOR. You transform HOW things are said, not WHAT is said.

The Writer has already crafted the analytical content. You MUST preserve:

1. PREDICTIONS - Keep all forward-looking language:
   - "will", "expect", "likely", "forecast", "anticipate"
   - Do NOT remove these to make prose "cleaner"

2. CONSTRAINTS - Keep actor limitation phrases:
   - "constrained by", "limited by", "forced to", "cannot", "unable to"
   - These are ANALYTICAL INSIGHTS, not filler

3. SHERMAN KENT TERMS - Keep probability language EXACTLY:
   - "HIGHLY LIKELY", "LIKELY", "UNLIKELY", etc.
   - Do NOT soften these or replace with vague alternatives

4. ENTITY DENSITY - Keep specific names, numbers, dates:
   - Do NOT replace "$4.2 billion" with "significant investment"
   - Do NOT replace "Tehran" with "the regime"
   - Specificity is NOT clutter—it IS the insight

5. CAUSAL LANGUAGE - Keep cause-effect chains:
   - "because", "therefore", "resulting in", "forcing"
   - These show analytical reasoning, not just description

BAD TRANSFORMATION (removes analysis):
- Writer: "Constrained by depleted reserves, Ankara is HIGHLY LIKELY to seek Gulf funding."
- Stylist: "Ankara's financial pressures may drive new partnerships."
  ❌ Lost: "Constrained by", "HIGHLY LIKELY", specific constraint

GOOD TRANSFORMATION (preserves analysis, improves style):
- Writer: "Constrained by depleted reserves, Ankara is HIGHLY LIKELY to seek Gulf funding."
- Stylist: "Depleted reserves leave Ankara little choice—Gulf funding is now HIGHLY LIKELY."
  ✓ Kept: constraint, probability term, causal logic
  ✓ Improved: sentence structure, rhythm

If you strip analytical content to make prose "flow better," you have FAILED.
</analytical_content_preservation>

<orwell_rules>
Apply these ruthlessly:
1. Never use a metaphor you've seen before
2. Never use a long word where a short one will do
3. If you can cut a word, cut it
4. Never use passive when active will work
5. Never use jargon when everyday words work
6. Break any rule rather than say something barbarous
</orwell_rules>

<banned_phrases>
These must NEVER appear:
- "remains to be seen"
- "only time will tell"
- "at the end of the day"
- "moving forward"
- "perfect storm"
- "game changer"
- "kick the can"
- "tip of the iceberg"
- Any corporate or diplomatic cliché
</banned_phrases>

<output_format>
Return the transformed prose as a single string.
Keep the same structure (paragraph breaks, etc.).
Only improve the language—do not add or remove facts.
</output_format>
"""


def _build_stylist_prompt(input_data: StylistInput) -> str:
    """Build the user prompt for the Stylist agent."""
    
    # Get relevant style examples
    # Determine topic from blueprint archetype
    topic_map = {
        "CRISIS": "Conflict",
        "TREND": "Economic",
        "PIVOT": "Diplomatic",
        "SLEEPER": "Economic",
        "COMPETITION": "Conflict",
        "CONSTRAINT": "Economic",
    }
    topic = topic_map.get(input_data.blueprint.archetype, "Economic")
    
    # Get 4 examples (more variety) instead of 2
    config = get_config()
    examples = get_style_examples(
        topic=topic,
        tone=input_data.tone,
        limit=4,
        preset=config.style_preset,
    )
    
    # Format sacred elements
    sacred_list = []
    if input_data.sacred_elements.proper_nouns:
        sacred_list.append(f"**Names:** {', '.join(input_data.sacred_elements.proper_nouns)}")
    if input_data.sacred_elements.statistics:
        sacred_list.append(f"**Numbers:** {', '.join(input_data.sacred_elements.statistics)}")
    if input_data.sacred_elements.dates:
        sacred_list.append(f"**Dates:** {', '.join(input_data.sacred_elements.dates)}")
    if input_data.sacred_elements.quotes:
        sacred_list.append(f"**Quotes:** {json.dumps(input_data.sacred_elements.quotes)}")
    
    sacred_section = "\n".join(sacred_list) if sacred_list else "None specified"
    
    # Get voice mode modifier
    voice_modifier = get_voice_modifier(input_data.voice_mode)
    voice_name = VOICE_MODES.get(input_data.voice_mode, VOICE_MODES["ECONOMIST_STRATFOR"])["description"]
    
    # Build revision section if this is a retry
    revision_section = ""
    if input_data.critic_feedback:
        previous_styled_block = ""
        if input_data.previous_styled:
            previous_styled_block = f"""
### YOUR PREVIOUS STYLING (that failed):

```markdown
{input_data.previous_styled}
```

"""
        revision_section = f"""
## ⚠️ REVISION REQUIRED

Your previous styling FAILED the Style Critic audit.

{previous_styled_block}
### CRITIC FEEDBACK:

{input_data.critic_feedback}

### YOUR TASK:

1. Read your previous styled version above
2. Understand each style issue the Critic identified
3. Restyle the ORIGINAL Writer draft below, fixing ALL issues
4. The Writer draft is your source of truth for FACTS - do NOT change facts

Do NOT just make minor edits. Transform with the feedback in mind.

---

"""

    prompt = f"""
## SECTION TO STYLE

**Region:** {input_data.region}
**Archetype:** {input_data.blueprint.archetype}
**Target Tone:** {input_data.tone}
**Voice Mode:** {voice_name}

{revision_section}
{voice_modifier}

## SACRED ELEMENTS (DO NOT ALTER)

{sacred_section}

## IMPORTANT: PRESERVE SOURCES SECTION

If the draft includes a **Sources:** section listing external references ([1], [2], etc.), 
you MUST preserve it exactly at the end of your transformed prose.

Do NOT remove, reformat, or "improve" the Sources section. Keep it verbatim.

## IMPORTANT: PRESERVE IMAGE MARKERS

If the draft includes any `<!-- IMAGE: ... -->` comments, you MUST preserve them exactly.
These are image generation markers used downstream. Copy them verbatim to your output.

## STYLE EXAMPLES

{examples}

## BAD EXAMPLE: What NOT to Do

**Draft (flat, generic):**
"The central bank's decision to raise interest rates was announced this week. This move is expected to have significant implications for the economy. Analysts say this could potentially impact inflation. Only time will tell if this strategy will be effective."

**Bad transformation (still flat):**
"The central bank decided to raise interest rates this week. This important move is likely to significantly affect the economy. Many experts believe this might help control inflation. The results remain to be seen."

**Problem:** Same structure, just swapped words. No imagery, no specifics, kept clichés ("significant," "remains to be seen"), no memorable lines. This is INCREMENTAL editing, not TRANSFORMATION.

---

✓ **Good transformation (actually transformed):**
"Four rate rises in eight months—and the central bank shows no signs of blinking. At 6.5%, borrowing costs have tripled since January. The strategy is textbook: crush demand before inflation expectations become entrenched. But textbooks rarely account for an election year."

**Why it works:** Front-loaded number, concrete image ("no signs of blinking"), specific stats, wry observation, varied sentence length.

---

## DRAFT TO TRANSFORM

```markdown
{input_data.writer_draft}
```

## YOUR TASK: TWO STEPS

### STEP 1: EDITORIAL THINKING (write this out first)

Before transforming, answer these 4 questions about the section. Actually write them.

1. **THE IRONY**: What's absurd, contradictory, or darkly funny about this situation?
   Every story has an irony. Find it. Name the contradiction.
   
2. **THE STANCE**: What's our editorial take? Not neutral—what do we actually think?
   Be opinionated. What's the truth beneath the surface?
   
3. **THE WRY LINE**: One sentence a seasoned analyst would chuckle at or quote.
   Should capture the essence with understated wit.
   
4. **THE IMAGE**: One concrete image or metaphor that makes this visual.
   Not abstract ("pressure increases") but concrete ("dropped the hammer").

---

### STEP 2: TRANSFORM THE PROSE

Now transform using your answers above:

- Let the **IRONY** inform your **TONE**
- Let the **STANCE** inform your **WORD CHOICE**
- The **WRY LINE** MUST appear somewhere (opening, body, or close)
- The **IMAGE** MUST appear somewhere (make it visual)

Additional requirements:
- Fix sentence variety (no more than 2 consecutive sentences starting the same)
- Use active voice (capitals as actors)
- Front-load specifics (numbers, dates) 
- Apply Orwell's rules (short words, cut filler)
- PRESERVE ALL SACRED ELEMENTS EXACTLY

---

## OUTPUT FORMAT

```
EDITORIAL THINKING:

**THE IRONY:** [your answer]

**THE STANCE:** [your answer]

**THE WRY LINE:** "[the specific quotable phrase]"

**THE IMAGE:** "[the specific concrete image/metaphor]"

---

TRANSFORMED PROSE:

[Your transformed draft here. Same structure, transformed voice.]
```

Return both sections. The editorial thinking helps me verify you found the insight, not just swapped words.
"""
    
    return prompt


# =============================================================================
# TRANSFORMATION VALIDATION
# =============================================================================

def _count_sentence_starts(text: str) -> dict[str, int]:
    """Count how sentences start to check for variety."""
    import re
    sentences = re.split(r'[.!?]\s+', text)
    starts = {}
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        words = sentence.strip().split()
        if words:
            first_word = words[0].rstrip(',;:')
            starts[first_word] = starts.get(first_word, 0) + 1
    
    return starts


def _check_transformation_quality(original: str, styled: str) -> tuple[list[str], list[str]]:
    """
    Check transformation quality for logging.
    
    Returns:
        (improvements_found, issues_found)
    """
    improvements = []
    issues = []
    
    # 1. Sentence variety
    original_starts = _count_sentence_starts(original)
    styled_starts = _count_sentence_starts(styled)
    
    max_original = max(original_starts.values()) if original_starts else 0
    max_styled = max(styled_starts.values()) if styled_starts else 0
    
    if max_styled < max_original:
        improvements.append(f"Sentence variety improved (max repetition: {max_original} → {max_styled})")
    elif max_styled >= 3:
        most_common = max(styled_starts, key=styled_starts.get)
        issues.append(f"Repetitive starts: '{most_common}' used {max_styled} times")
    
    # 2. Cliché removal
    original_cliches = check_for_cliches(original)
    styled_cliches = check_for_cliches(styled)
    
    if len(styled_cliches) < len(original_cliches):
        improvements.append(f"Removed {len(original_cliches) - len(styled_cliches)} clichés")
    if styled_cliches:
        issues.append(f"Clichés remain: {styled_cliches[:2]}")
    
    # 3. Word count check
    original_words = len(original.split())
    styled_words = len(styled.split())
    word_change_pct = ((styled_words - original_words) / original_words * 100)
    
    if abs(word_change_pct) <= 15:
        improvements.append(f"Word count maintained ({word_change_pct:+.0f}%)")
    elif styled_words > original_words * 1.2:
        issues.append(f"Word count increased {word_change_pct:+.0f}%")
    
    # 4. Check for memorable elements (heuristic)
    has_em_dash = '—' in styled
    has_question = '?' in styled
    has_colon = ':' in styled
    short_sentences = len([s for s in styled.split('.') if 3 <= len(s.split()) <= 8])
    
    quotable_score = sum([has_em_dash, has_question, has_colon, short_sentences >= 1])
    if quotable_score >= 2:
        improvements.append("Contains quotable elements")
    
    return improvements, issues


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def _parse_stylist_response(response_text: str) -> tuple[dict, str]:
    """
    Parse Stylist response into editorial thinking and transformed prose.
    
    Returns:
        (editorial_thinking_dict, transformed_prose_string)
    """
    import re
    
    editorial_thinking = {}
    transformed_prose = response_text  # Fallback
    
    # Try to split on "TRANSFORMED PROSE:" or similar markers
    split_markers = [
        "TRANSFORMED PROSE:",
        "## TRANSFORMED PROSE",
        "TRANSFORMED:",
        "---\n\n",  # The separator in our format
    ]
    
    prose_start_idx = -1
    for marker in split_markers:
        if marker in response_text:
            prose_start_idx = response_text.index(marker) + len(marker)
            break
    
    # Extract editorial thinking section
    if prose_start_idx > 0:
        thinking_section = response_text[:prose_start_idx]
        transformed_prose = response_text[prose_start_idx:].strip()
        
        # Parse the 4 key elements from thinking section
        # THE IRONY
        irony_match = re.search(r'\*\*THE IRONY:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', thinking_section, re.DOTALL)
        if irony_match:
            editorial_thinking['irony'] = irony_match.group(1).strip()
        
        # THE STANCE
        stance_match = re.search(r'\*\*THE STANCE:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', thinking_section, re.DOTALL)
        if stance_match:
            editorial_thinking['stance'] = stance_match.group(1).strip()
        
        # THE WRY LINE
        wry_match = re.search(r'\*\*THE WRY LINE:\*\*\s*["\']?(.+?)["\']?(?=\n\*\*|\n\n|$)', thinking_section, re.DOTALL)
        if wry_match:
            editorial_thinking['wry_line'] = wry_match.group(1).strip().strip('"\'')
        
        # THE IMAGE
        image_match = re.search(r'\*\*THE IMAGE:\*\*\s*["\']?(.+?)["\']?(?=\n\*\*|\n\n|$)', thinking_section, re.DOTALL)
        if image_match:
            editorial_thinking['image'] = image_match.group(1).strip().strip('"\'')
    
    # Clean up transformed prose (remove markdown code fences if present)
    transformed_prose = transformed_prose.strip()
    if transformed_prose.startswith('```'):
        # Remove opening fence
        lines = transformed_prose.split('\n')
        transformed_prose = '\n'.join(lines[1:])
    if transformed_prose.endswith('```'):
        # Remove closing fence
        transformed_prose = transformed_prose.rsplit('```', 1)[0]
    
    transformed_prose = transformed_prose.strip()
    
    return editorial_thinking, transformed_prose


# =============================================================================
# STYLIST AGENT
# =============================================================================

@with_retry(max_attempts=3, initial_delay=2.0, max_delay=30.0)
async def run_stylist_agent(
    input_data: StylistInput,
    client: genai.Client | None = None,
) -> str:
    """
    Run the Stylist Agent to transform prose.

    Args:
        input_data: Writer draft and context
        client: Optional Gemini client (created if not provided)

    Returns:
        Transformed prose string
    """
    config = get_config()

    if client is None:
        client = genai.Client(api_key=config.gemini_api_key)

    model = config.models.stylist
    logger.info(f"   ✨ [{input_data.region}] Running Stylist Agent with {model}...")

    prompt = _build_stylist_prompt(input_data)

    # Configure for text output - only set params if explicitly configured
    gen_config = types.GenerateContentConfig(
        system_instruction=STYLIST_SYSTEM_PROMPT,
    )
    
    # Only set temperature if explicitly configured
    if config.stylist_temperature is not None:
        gen_config.temperature = config.stylist_temperature
    
    # Only set max_output_tokens if explicitly configured
    if config.stylist_max_output_tokens is not None:
        gen_config.max_output_tokens = config.stylist_max_output_tokens

    # Log config for debugging
    log_model_config("Stylist", model, gen_config)

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=gen_config,
    )

    result = response.text.strip()
    
    # Parse editorial thinking and extract transformed prose
    editorial_thinking, transformed_prose = _parse_stylist_response(result)
    
    # Defensive: Restore Sources section if Stylist removed it
    if "**Sources:**" in input_data.writer_draft and "**Sources:**" not in transformed_prose:
        # Extract Sources section from original draft
        draft_parts = input_data.writer_draft.split("**Sources:**")
        if len(draft_parts) > 1:
            sources_section = "**Sources:**" + draft_parts[1]
            transformed_prose = transformed_prose.rstrip() + "\n\n" + sources_section
            logger.info(f"   🔗 [{input_data.region}] Restored Sources section removed by Stylist")
    
    # Log editorial insights
    if editorial_thinking:
        logger.info(f"   💡 [{input_data.region}] Editorial thinking:")
        if editorial_thinking.get('irony'):
            logger.info(f"      Irony: {editorial_thinking['irony'][:80]}...")
        if editorial_thinking.get('wry_line'):
            logger.info(f"      Quotable: \"{editorial_thinking['wry_line']}\"")
        if editorial_thinking.get('image'):
            logger.info(f"      Image: \"{editorial_thinking['image']}\"")
    else:
        logger.warning(f"   ⚠️ [{input_data.region}] No editorial thinking section found")
    
    # Strip any leaked metadata labels (defensive)
    transformed_prose = strip_metadata_labels(transformed_prose, region=input_data.region)
    
    # Post-process with Orwell Filter
    transformed_prose = apply_orwell_filter(transformed_prose)
    
    # Check transformation quality
    improvements, issues = _check_transformation_quality(
        original=input_data.writer_draft,
        styled=transformed_prose
    )
    
    # Log quality assessment
    if improvements:
        logger.info(f"   ✓ [{input_data.region}] Transformations: {', '.join(improvements[:3])}")
    if issues:
        logger.warning(f"   ⚠️ [{input_data.region}] Issues: {', '.join(issues[:3])}")
    
    # Verify sacred elements preserved
    violations = verify_sacred_elements(
        transformed_prose, 
        input_data.sacred_elements,
        input_data.writer_draft,
    )
    if violations:
        logger.warning(
            f"   ⚠️ [{input_data.region}] Sacred element violations: {violations[:2]}"
        )
    
    logger.info(
        f"   ✅ [{input_data.region}] Stylist complete: "
        f"{len(transformed_prose.split())} words"
    )
    
    return transformed_prose


# =============================================================================
# ORWELL FILTER (POST-PROCESSING)
# =============================================================================

def apply_orwell_filter(text: str) -> str:
    """
    Apply Orwell-style improvements to text.
    
    - Replaces long words with short ones
    - Removes clichés
    """
    # Apply brevity substitutions
    result = apply_brevity_substitutions(text)
    
    # Remove known clichés (replace with empty or simpler phrase)
    cliche_replacements = {
        "at the end of the day": "",
        "moving forward": "",
        "remains to be seen": "is unclear",
        "only time will tell": "the outcome is uncertain",
        "perfect storm": "convergence",
        "game changer": "significant shift",
        "kick the can down the road": "delay",
        "tip of the iceberg": "just the beginning",
    }
    
    for cliche, replacement in cliche_replacements.items():
        if cliche.lower() in result.lower():
            # Case-insensitive replacement
            import re
            pattern = re.compile(re.escape(cliche), re.IGNORECASE)
            result = pattern.sub(replacement, result)
    
    # Clean up any double spaces
    while "  " in result:
        result = result.replace("  ", " ")
    
    return result.strip()


# =============================================================================
# SACRED ELEMENTS VERIFICATION
# =============================================================================

def verify_sacred_elements(
    styled_text: str,
    sacred: SacredElements,
    original_text: str,
) -> list[str]:
    """
    Verify that sacred elements are preserved in styled text.
    
    Returns:
        List of violations (empty if all preserved)
    """
    violations = []
    
    # Check proper nouns
    for noun in sacred.proper_nouns:
        if noun in original_text and noun not in styled_text:
            violations.append(f"Missing proper noun: {noun}")
    
    # Check statistics
    for stat in sacred.statistics:
        if stat in original_text and stat not in styled_text:
            violations.append(f"Missing statistic: {stat}")
    
    # Check dates
    for date in sacred.dates:
        if date in original_text and date not in styled_text:
            violations.append(f"Missing date: {date}")
    
    # Check quotes (exact match required)
    for quote in sacred.quotes:
        if quote in original_text and quote not in styled_text:
            violations.append(f"Modified quote: {quote[:30]}...")
    
    return violations


# =============================================================================
# SACRED ELEMENTS EXTRACTION
# =============================================================================

def extract_sacred_elements(
    analyst_output: "AnalystOutput",
    events: list[dict],
) -> SacredElements:
    """
    Extract sacred elements from analyst output and source events.
    
    Args:
        analyst_output: The structured analysis
        events: Raw source events
        
    Returns:
        SacredElements that must be preserved
    """
    import re
    
    proper_nouns = set()
    statistics = set()
    dates = set()
    quotes = []
    event_ids = []
    
    # Extract from analyst output
    for actor in analyst_output.primary_actors:
        proper_nouns.add(actor.actor)
    
    # Extract from futures wheel
    event_ids.append(analyst_output.futures_wheel.driver_event_id)
    
    # Extract from events
    for event in events[:20]:
        # Get event ID
        if event.get("id"):
            event_ids.append(event["id"])
        
        # Extract entities from title/summary
        text = f"{event.get('title', '')} {event.get('summary', '')}"
        
        # Find numbers/statistics (patterns like $100B, 15%, 1.5 million)
        stat_patterns = [
            r'\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion|B|M|T))?',
            r'\d+(?:\.\d+)?%',
            r'[\d,]+\s*(?:billion|million|trillion)',
        ]
        for pattern in stat_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            statistics.update(matches)
        
        # Find dates (patterns like January 15, March 2026, etc.)
        date_patterns = [
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?',
            r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{4})?',
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.update(matches)
    
    return SacredElements(
        proper_nouns=list(proper_nouns),
        statistics=list(statistics),
        dates=list(dates),
        quotes=quotes,
        event_ids=event_ids,
    )


# =============================================================================
# BATCH PROCESSING
# =============================================================================

async def run_stylist_agents_parallel(
    inputs: list[StylistInput],
    client: genai.Client | None = None,
) -> dict[str, str]:
    """
    Run Stylist agents in parallel for all sections.
    
    Args:
        inputs: List of StylistInput for each section
        client: Shared Gemini client
        
    Returns:
        Dict mapping region to styled prose
    """
    import asyncio
    
    logger.info(f"   ✨ Running {len(inputs)} Stylist agents in parallel...")
    
    async def run_with_fallback(input_data: StylistInput) -> tuple[str, str]:
        try:
            result = await run_stylist_agent(input_data, client)
            return input_data.region, result
        except Exception as e:
            logger.error(f"   ❌ [{input_data.region}] Stylist failed: {e}")
            # Return original draft as fallback
            return input_data.region, input_data.writer_draft
    
    results = await asyncio.gather(*[run_with_fallback(inp) for inp in inputs])
    return dict(results)


# =============================================================================
# SYNC WRAPPER
# =============================================================================

def run_stylist_agent_sync(
    input_data: StylistInput,
    client: genai.Client | None = None,
) -> str:
    """Synchronous wrapper for run_stylist_agent."""
    import asyncio
    return asyncio.run(run_stylist_agent(input_data, client))
