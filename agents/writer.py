"""
Writer Agent - Prose Layer

This agent transforms STRUCTURED ANALYTICAL DATA into high-density prose.
It implements:
- Bottom-line-up-front format (lead with key judgment)
- Chain of Density (CoD) for information-rich writing (3-5 iterative passes)
- Sherman Kent probability language
- Entity-dense prose (names, numbers, dates)
- SectionBlueprint integration for structured paragraph plans

The input is AnalystOutput JSON, NOT raw events.
The output is polished markdown ready for publication.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Optional

from google import genai
from google.genai import types

from config import get_config
from utils import logger, with_retry, log_model_config
from .schemas import AnalystOutput, WriterInput, WriterOutput

if TYPE_CHECKING:
    from ..state import SectionBlueprint, SacredElements


# =============================================================================
# WRITER PROMPTS
# =============================================================================

WRITER_SYSTEM_PROMPT = """
<role>
You are Editor-in-Chief of a premium geopolitical intelligence briefing.
Your job is COMMUNICATION, not analysis. The analysis is already done.
You transform structured analytical data into compelling, dense prose.
</role>

<audience>
You write for senior decision-makers who need to:
1. Sound smart in meetings (agility)
2. Take action based on your analysis (agency)
3. Scan in 30 seconds OR read deeply (dual-speed)
</audience>

<constraints>
BANNED PHRASES (instant failure if used):
- "significant developments"
- "remains to be seen"
- "time will tell"
- "various factors"
- "could potentially"
- "the situation is evolving"
- "in the coming weeks"
- "major implications"

PROHIBITED:
- Chronological narration ("First X happened, then Y")
- Passive voice ("It was reported that")
- Personality attribution ("Putin's ambition", "Xi's desire")
- Weasel words
- Hedging without probability terms
- Summary without insight
</constraints>

<opening_judgment>
Lead with your key judgment (bottom-line-up-front, but don't use that as a header).

Every section starts with a KEY JUDGMENT combining:
- The PRIMARY CONSTRAINT limiting the main actor
- The 3RD ORDER EFFECT (the "So What?")

Template:
"Constrained by [CONSTRAINT], [ACTOR] is [PROBABILITY] to [ACTION], likely resulting in [3RD ORDER EFFECT]."

Example:
"Constrained by depleted foreign reserves and IMF conditionality, Ankara is HIGHLY LIKELY (80-90%) to delay further rate cuts, forcing a pivot toward Gulf sovereign wealth funds that will deepen Turkey's strategic dependency on Riyadh."
</opening_judgment>

<chain_of_density>
Your prose must be information-dense:
- Every sentence contains at least one specific entity
- Entities: Dates, dollar amounts, locations, weapon systems, proper nouns
- NO filler phrases whatsoever
</chain_of_density>

<sherman_kent_scale>
Use ONLY these probability terms:
- ALMOST CERTAIN (93-99%)
- HIGHLY LIKELY (80-92%)
- LIKELY (60-79%)
- ROUGHLY EVEN (40-59%)
- UNLIKELY (21-39%)
- HIGHLY UNLIKELY (8-20%)
- REMOTE (1-7%)
</sherman_kent_scale>

<metonyms>
Use metonyms for state actors (emphasizes interests over personalities):
- "Washington" not "Biden" or "the US government"
- "Beijing" not "Xi" or "China"
- "The Kremlin" not "Putin"
- "Ankara" not "Erdogan"
</metonyms>

<section_structures>

FEATURED ANALYSIS (400-600 words):
1. Opening hook with Key Judgment (lead with main finding)
2. Geopolitical archetype explanation (why this pattern matters)
3. Constraints analysis (what actors CANNOT do)
4. Futures wheel integration (1st → 2nd → 3rd order effects)
5. Contradicting evidence / dissent (what argues against you)
6. Outlook with Sherman Kent probability

REGIONAL BRIEFING (200-300 words):
1. Key Judgment (lead with main finding)
2. Primary constraint
3. Second-order effect
4. Contradicting evidence (1 sentence)
5. Outlook

SLEEPER SECTION (80-100 words max):
A quiet development that could become significant later.
Use "watch this" framing — plant a seed for future coverage.
Structure:
1. Surface calm (what appears normal) — 1 sentence
2. The anomaly beneath (what's actually happening) — 1-2 sentences
3. Activation trigger (what would make this blow up) — 1 sentence
4. Watch for (specific indicator to monitor) — 1 sentence
Tone: Intriguing, slightly mysterious. Like a reporter's tip.

</section_structures>

<sourcing_rules>
This briefing is AI-generated. Users MUST be able to fact-check every claim.

PRIMARY SOURCES (Event Database):
**MANDATORY FORMAT:** [short phrase](https://realpolitik.world/?event=EVENT_ID)

**CRITICAL:** Link text must be SHORT - ideally 1-4 words, maximum 6 words.
Link the KEY NOUN or PHRASE, not entire sentences.

EXCELLENT (1-4 words):
- "Tehran's [brutal crackdown](https://realpolitik.world/?event=abc123) killed thousands."
- "Washington deployed an [armada](https://realpolitik.world/?event=def456) to the Gulf."
- "The [Greenland acquisition bid](https://realpolitik.world/?event=xyz789) shocked allies."
- "Maduro was [reportedly kidnapped](https://realpolitik.world/?event=123abc) by US forces."

ACCEPTABLE (5-6 words):
- "The [10% tariff threat against Europe](https://realpolitik.world/?event=abc) escalated."

FORBIDDEN (too long, clunky):
❌ "Tehran is facing [widespread protests amid internet blackouts and military crackdowns](url)."
❌ "Washington's [capture of Maduro which shocked the international community](url) occurred."
❌ "Recent [tariff threats over Greenland sovereignty and proposed Gaza Board of Peace](url)..."

FORBIDDEN (technical errors):
❌ [7d864e9aee084ab9] ← Just an ID, no descriptive text
❌ https://realpolitik.world/?event=xyz ← Bare URL

MULTIPLE EVENTS IN ONE SENTENCE:
Link each separately:
"Trump's [Greenland bid](event1) and [Gaza initiative](event2) reshaped alliances."
NOT: "Trump's [Greenland bid and Gaza initiative](event1) reshaped alliances."

EXTERNAL SOURCES (Tavily/Web Search):
If the Analyst used external sources from web search, they will be listed below.
Cite them using numbered references inline:

CORRECT:
- "The death toll exceeded 5,000 [1]."
- "According to recent assessments [2], the regime faces collapse."
- "Multiple reports [1][3] confirm the deployment."

Then add a **Sources** section at the END of your section:
**Sources:**  
[1] BBC News, "Iran Protests Escalate", January 20, 2026  
[2] Reuters, "Tehran Regime Under Pressure", January 21, 2026

CITATION DISCIPLINE:
- Internal events (realpolitik.world): SHORT inline links (1-4 words)
- External sources (Tavily): [1], [2] style inline, listed at section end
- One link per event - don't bundle multiple events into one link
- Keep all link text under 6 words for readability
</sourcing_rules>

<output_format>
OUTPUT CLEAN PROSE ONLY. No structural metadata headers.

DO NOT include these labels in your prose (they are for YOUR reference only):
- "BLUF"
- Archetype names: "CRISIS", "TREND", "PIVOT", "SLEEPER", "COMPETITION", "CONSTRAINT"
- "SECTION:" or region names as headers (the document assembler adds these)
- "**AMERICAS**", "**MIDDLE_EAST**", "**CHINA**" or any region tags
- Hook/closing types: "STATEMENT", "QUESTION", "SURPRISE", "CONTRAST", "PREDICTION", "WARNING"
- Spoke labels: "SPOKE 1", "SPOKE 2", "HUB"
- Blueprint markers: "PARAGRAPH 1:", "BEAT:", "OPENING:", "CLOSING:"

Your output should be publication-ready prose with inline event citations.
Return polished markdown prose following the section structure for the given section type.
</output_format>
"""


def _get_section_structure(section_type: str, blueprint: Optional["SectionBlueprint"] = None) -> str:
    """Get the structure template for the section type."""
    
    # If we have a blueprint from Structure agent, use it
    if blueprint:
        return _format_blueprint_structure(blueprint)
    
    # Fallback to default templates
    if section_type == "featured":
        return """
## FEATURED ANALYSIS: {Use the archetype or key theme as title}

<!-- IMAGE: Brief visual concept for this section - describe the key tension or dynamic in 10-15 words -->

[400-600 words following this structure:]

**Opening paragraph**: Key Judgment (lead with main finding). This is your thesis statement.

**Archetype paragraph**: Explain the geopolitical framework at play. Why does this pattern matter historically?

**Constraints paragraph**: What are actors FORCED to do? Use the Const-o-T analysis.

**Causal chain paragraph**: Walk through 1st → 2nd → 3rd order effects. The 3rd order is your "So What?"

**Dissent paragraph**: Acknowledge contradicting evidence. What could prove you wrong?

**Outlook**: Final prediction with Sherman Kent probability term and specific timeframe.

**Sources**: If you used external search, list [1], [2], etc. here. Otherwise, all sources are event links above.
"""
    elif section_type == "sleeper":
        return """
## 🔮 Sleeper Watch: {REGION}

[80-100 words MAX — this is a seed, not an analysis]

**Surface calm**: What appears normal. One sentence.

**The anomaly**: What's quietly happening beneath the surface. 1-2 sentences with specific detail.

**Activation trigger**: What would make this blow up? One sentence.

**Watch for**: Specific indicator to monitor. One sentence.

Tone: Intriguing, slightly mysterious. Like a reporter's tip.
Do NOT write a full analysis. Plant a seed for future coverage.
"""
    else:
        return """
## {REGION} — {TREND}

[200-300 words following this structure:]

**KEY JUDGMENT**: Lead with main finding, combining constraint + 3rd order effect.

**Analysis**: Primary constraint and second-order effect. Be specific with entities. Cite sources with event links or [n] notation.

**Contradicting evidence**: One sentence acknowledging what argues against your assessment.

**Outlook**: Sherman Kent probability term with specific prediction.
"""


def _format_blueprint_structure(blueprint: "SectionBlueprint") -> str:
    """
    Format a SectionBlueprint into prose guidance for the Writer.
    
    CRITICAL: Translates structural metadata into guidance WITHOUT exposing labels.
    The Writer should never see "CRISIS" or "STATEMENT" as headers to include.
    """
    lines = []
    
    # Translate archetype to tone guidance (don't show archetype label)
    archetype_guidance = {
        "CRISIS": "Write with urgency. Focus on immediate stakes and constraints.",
        "TREND": "Write analytically. Show the pattern emerging over time.",
        "PIVOT": "Write about trajectory shift. Emphasize before/after contrast.",
        "SLEEPER": "Write with understated significance. This matters more than it seems.",
        "COMPETITION": "Write about rival strategies and zero-sum dynamics.",
        "CONSTRAINT": "Write about limits. Show what actors CANNOT do.",
        "LEADER": "Write with authority about the week's defining development.",
    }
    
    tone = archetype_guidance.get(blueprint.archetype, "Write clearly and analytically.")
    
    # NOTE: Don't include SECTION header - assembler adds it. Just provide context.
    lines.append(f"**Tone:** {tone}")
    lines.append(f"**Target:** {blueprint.word_target} words")
    lines.append(f"**Context:** Writing about {blueprint.region}")
    lines.append("")
    
    # Hook guidance (translate type to instruction, don't show label)
    hook_instructions = {
        "STATEMENT": "Open with a bold declarative statement.",
        "QUESTION": "Open with a provocative question.",
        "SURPRISE": "Open with an unexpected fact or reversal.",
        "CONTRAST": "Open with a before/after or ironic contrast.",
        "SCENE": "Open with a specific moment or image.",
    }
    
    hook_guide = hook_instructions.get(blueprint.hook_type, "Open strongly.")
    
    lines.append(f"**OPENING:** {hook_guide}")
    lines.append(f"Hook inspiration: \"{blueprint.hook_draft}\"")
    lines.append("")
    
    # Paragraph-by-paragraph plan
    lines.append("**PARAGRAPH STRUCTURE:**")
    lines.append("")
    for i, para in enumerate(blueprint.paragraphs, 1):
        key_facts = ", ".join(para.key_facts[:3]) if para.key_facts else "relevant facts"
        lines.append(f"{i}. **{para.purpose}** (~{para.word_target} words)")
        lines.append(f"   Beat: {para.beat}")
        lines.append(f"   Key facts: {key_facts}")
        lines.append("")
    
    # Key events emphasis
    if blueprint.key_events:
        lines.append("**KEY EVENTS TO EMPHASIZE:**")
        for event in blueprint.key_events:
            title = event.event_title or event.event_id
            role = event.role or event.how_to_use or "integrate naturally"
            emphasis_label = {"LEAD": "Primary", "SUPPORT": "Supporting", "MENTION": "Mention"}.get(event.emphasis, event.emphasis)
            lines.append(f"- [{emphasis_label}] {title}: {role}")
        lines.append("")
    
    # Closing guidance (translate type to instruction, don't show label)
    closing_instructions = {
        "PREDICTION": "End with a forward-looking prediction using Sherman Kent probability.",
        "WARNING": "End with a cautionary judgment about risks.",
        "QUESTION": "End with a provocative question that lingers.",
        "CALLBACK": "End by echoing the opening hook or image.",
        "IMAGE": "End with a concrete image or memorable phrase.",
    }
    
    closing_guide = closing_instructions.get(blueprint.closing_type, "End strongly.")
    
    lines.append(f"**CLOSING:** {closing_guide}")
    lines.append(f"Ending inspiration: \"{blueprint.closing_draft}\"")
    lines.append("")
    
    # Add image intent instruction for featured sections
    # Writer should place it after the headline
    lines.append("**IMAGE INTENT (for featured/major sections):**")
    lines.append("After your headline, add:")
    lines.append("<!-- IMAGE: Describe the key visual in 10-15 words -->")
    lines.append("Example: <!-- IMAGE: Energy infrastructure under attack, darkened power grid, winter atmosphere -->")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# CHAIN OF DENSITY IMPLEMENTATION
# =============================================================================


# =============================================================================
# POST-PROCESSING HELPERS
# =============================================================================

def strip_metadata_labels(text: str, region: str = "") -> str:
    """
    Remove structural labels that leaked into prose.
    
    This is a defensive safety net - if this fires, it means the prompt
    guidance isn't working and should be improved.
    """
    import re
    
    original = text
    lines = text.split('\n')
    cleaned_lines = []
    
    # Patterns for metadata headers that shouldn't appear in prose
    # IMPORTANT: Be specific to avoid stripping valid content like **Sources:**
    metadata_patterns = [
        r'^##?\s*BLUF\s*$',  # ## BLUF or # BLUF (standalone)
        r'^\*\*BLUF\s*\([^)]+\)\s*:\*\*',  # **BLUF (Bottom Line Up Front):**
        r'^##\s+.+?\s+—\s+(?:CRISIS|TREND|PIVOT|SLEEPER|COMPETITION|CONSTRAINT|LEADER)\s*$',  # ## Region — ARCHETYPE
        r'^##?\s*SPOKE\s+\d+\s*$',  # ## SPOKE 1
        r'^##?\s*(?:CRISIS|TREND|PIVOT|SLEEPER|COMPETITION|CONSTRAINT|LEADER)\s*$',  # ## CRISIS (standalone archetype header)
        r'^\s*—+\s*(?:CRISIS|TREND|PIVOT|SLEEPER|COMPETITION|CONSTRAINT)\s*$',  # — CRISIS
        r'^##?\s*HUB\s*$',  # ## HUB
        r'^##?\s*(?:OPENING|CLOSING)\s*$',  # ## OPENING or ## CLOSING
        r'^\*\*(?:OPENING|CLOSING)\s*\([A-Z]+\)\s*:\*\*',  # **OPENING (STATEMENT):**
        r'^\s*PARAGRAPH\s+\d+\s*:',  # PARAGRAPH 1:
        r'^\s*BEAT\s*:',  # BEAT:
        r'^\*\*(?:KEY EVENTS|PARAGRAPH STRUCTURE)\s*:\*\*',  # **KEY EVENTS:** or **PARAGRAPH STRUCTURE:**
    ]
    
    stripped_count = 0
    
    for line in lines:
        # Check if this line is a metadata label
        is_metadata = any(re.match(pattern, line.strip(), re.IGNORECASE) for pattern in metadata_patterns)
        
        if is_metadata:
            stripped_count += 1
            # Skip this line
            continue
        else:
            cleaned_lines.append(line)
    
    cleaned = '\n'.join(cleaned_lines)
    
    if stripped_count > 0:
        logger.warning(
            f"   ⚠️ [{region}] Stripped {stripped_count} metadata labels from output "
            f"(prompt guidance may need improvement)"
        )
    
    return cleaned


# =============================================================================
# CHAIN OF DENSITY HELPERS
# =============================================================================

def _count_entities(text: str) -> int:
    """
    Count salient entities in text for density tracking.
    
    Counts:
    - Numbers (including percentages, dates)
    - Proper nouns (capitalized words not at sentence start)
    - Technical terms
    """
    import re
    
    entity_count = 0
    
    # Count numbers (including percentages, dollar amounts)
    numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?%?\b|\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|trillion))?\b', text)
    entity_count += len(numbers)
    
    # Count dates (Month Day, Year patterns)
    dates = re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s+\d{4})?\b', text)
    entity_count += len(dates)
    
    # Count proper nouns (capitalized words not at sentence start)
    # Simple heuristic: word is capitalized and not after period/start
    words = text.split()
    for i, word in enumerate(words):
        # Skip first word and words after sentence-ending punctuation
        if i == 0:
            continue
        if i > 0 and words[i-1].rstrip('.,!?;:').endswith(('.', '!', '?')):
            continue
        
        # Count if capitalized (likely proper noun)
        if word and word[0].isupper() and len(word) > 1:
            entity_count += 1
    
    return entity_count


def _find_missing_entities(draft: str, sacred_elements: "SacredElements") -> str:
    """
    Find sacred elements not yet mentioned in the draft.
    
    Returns formatted string of missing entities for CoD prompt.
    """
    missing = []
    
    # Check proper nouns
    if sacred_elements.proper_nouns:
        draft_lower = draft.lower()
        missing_nouns = [n for n in sacred_elements.proper_nouns if n.lower() not in draft_lower]
        if missing_nouns:
            missing.append(f"Names/Places: {', '.join(missing_nouns[:5])}")
    
    # Check statistics
    if sacred_elements.statistics:
        missing_stats = [s for s in sacred_elements.statistics if s not in draft]
        if missing_stats:
            missing.append(f"Numbers: {', '.join(missing_stats[:5])}")
    
    # Check dates
    if sacred_elements.dates:
        missing_dates = [d for d in sacred_elements.dates if d not in draft]
        if missing_dates:
            missing.append(f"Dates: {', '.join(missing_dates[:3])}")
    
    if missing:
        return "\n".join(f"- {item}" for item in missing)
    return ""


def _extract_improved_draft(response_text: str) -> str:
    """
    Robustly extract the improved draft from CoD response.
    
    Handles various response formats and removes metadata sections.
    """
    text = response_text.strip()
    
    # Try to find the improved draft section
    markers = [
        "## IMPROVED DRAFT",
        "IMPROVED DRAFT:",
        "IMPROVED DRAFT",
        "## Improved Draft",
        "Improved Draft:",
    ]
    
    for marker in markers:
        if marker in text:
            # Split on marker and take everything after
            parts = text.split(marker, 1)
            if len(parts) > 1:
                draft = parts[1].strip()
                
                # Remove any trailing metadata markers
                for end_marker in ["##", "---", "```"]:
                    if end_marker in draft:
                        draft = draft.split(end_marker)[0].strip()
                
                return draft
    
    # Fallback: try to extract the largest paragraph block
    # Split on double newlines and take the longest section
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if paragraphs:
        # Filter out obvious metadata (starts with ## or - )
        content_paragraphs = [p for p in paragraphs if not p.startswith('##') and not p.startswith('-')]
        if content_paragraphs:
            # Return the longest content paragraph
            return max(content_paragraphs, key=len)
    
    # Last resort: return the whole response
    logger.warning("Could not parse CoD response structure, using full text")
    return text


# =============================================================================
# CHAIN OF DENSITY PROMPT
# =============================================================================

CHAIN_OF_DENSITY_PROMPT = """You are performing Chain of Density iteration {iteration} of {total}.

## CURRENT DRAFT ({word_count} words)

{current_draft}

## TASK

Identify 1-3 missing salient entities that should be included:
- Specific names (people, organizations, places)
- Numbers (dates, dollar amounts, percentages, quantities)
- Technical terms (weapon systems, treaties, specific policies)

Then REWRITE the draft to include these entities WITHOUT increasing the word count.

**CRITICAL CONSTRAINTS:**
- Target word count: {word_count} words (±10% maximum)
- Compress abstract prose to make room for new entities
- Replace vague language ("significant", "recent", "major") with specific data
- Do NOT fabricate entities - only add facts from source
- Preserve ALL Sherman Kent probability terms exactly

## ENTITIES TO ADD
(List 1-3 specific entities you're adding from the source)

## IMPROVED DRAFT
(Rewrite with same or fewer words, higher entity density)
"""


async def _apply_chain_of_density(
    draft: str,
    client: genai.Client,
    model: str,
    iterations: int = 3,
    sacred_elements: "SacredElements | None" = None,
) -> str:
    """
    Apply iterative Chain of Density to increase information density.
    
    Each iteration:
    1. Identifies missing salient entities from sacred elements
    2. Rewrites to include them without increasing length
    3. Compresses abstract prose into specific facts
    
    Includes convergence detection and word count enforcement.
    """
    config = get_config()  # Get config for temperature/token settings
    
    current = draft
    previous_draft = draft
    original_word_count = len(draft.split())
    
    # Track entity density for convergence
    previous_entity_count = _count_entities(current)
    
    for i in range(iterations):
        # Build source entities hint if available
        source_hint = ""
        if sacred_elements:
            missing_entities = _find_missing_entities(current, sacred_elements)
            if missing_entities:
                source_hint = f"\n## SOURCE ENTITIES NOT YET IN DRAFT\n{missing_entities}\n"
            else:
                logger.info(f"   CoD pass {i+1}: All sacred elements already included, stopping")
                break
        
        prompt = CHAIN_OF_DENSITY_PROMPT.format(
            iteration=i + 1,
            total=iterations,
            word_count=len(current.split()),
            current_draft=current,
        ) + source_hint
        
        gen_config = types.GenerateContentConfig(
            system_instruction="You are an editor focused on information density. Add specific facts, remove filler.",
        )
        
        # Only set temperature if explicitly configured
        if config.writer_temperature is not None:
            gen_config.temperature = config.writer_temperature
        
        # Only set max_output_tokens if explicitly configured
        if config.writer_cod_max_tokens is not None:
            gen_config.max_output_tokens = config.writer_cod_max_tokens
        
        # Set thinking level for CoD (default OFF - simple rewrite doesn't need thinking)
        if config.writer_cod_thinking_level is not None:
            gen_config.thinking_config = types.ThinkingConfig(
                thinking_level=config.writer_cod_thinking_level,
            )
        
        # Log config for debugging
        log_model_config(f"Writer-CoD-Pass{i+1}", model, gen_config)
        
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=gen_config,
            )
            
            # Extract the improved draft from the response
            next_draft = _extract_improved_draft(response.text)
            
            # Word count enforcement (allow 15% tolerance)
            new_word_count = len(next_draft.split())
            if new_word_count > original_word_count * 1.15:
                logger.warning(
                    f"   CoD pass {i+1}: Word count exceeded ({new_word_count} vs {original_word_count}), "
                    f"reverting to previous iteration"
                )
                break
            
            # Convergence detection - stop if entity density plateaus
            new_entity_count = _count_entities(next_draft)
            entity_gain = new_entity_count - previous_entity_count
            
            logger.debug(
                f"   CoD pass {i+1}: {original_word_count} → {new_word_count} words, "
                f"{previous_entity_count} → {new_entity_count} entities (+{entity_gain})"
            )
            
            if entity_gain <= 0:
                logger.info(f"   CoD converged at iteration {i+1} (no entity gain)")
                break
            
            # Update for next iteration
            previous_draft = current
            current = next_draft
            previous_entity_count = new_entity_count
            
        except Exception as e:
            logger.warning(f"   CoD pass {i+1} failed: {e}, using previous iteration")
            break
    
    final_word_count = len(current.split())
    final_entity_count = _count_entities(current)
    initial_entity_count = _count_entities(draft)
    
    logger.info(
        f"   ✅ CoD complete: {original_word_count} → {final_word_count} words, "
        f"{initial_entity_count} → {final_entity_count} entities"
    )
    
    return current


def _get_angle_instructions(angle: str) -> str:
    """
    Get specific execution instructions based on the angle type.
    
    Maps angle type to concrete "how to write it" guidance.
    """
    angle_upper = angle.upper()
    
    if "MISSING_THE_POINT" in angle_upper or "MISSING THE POINT" in angle_upper:
        return """
**HOW TO EXECUTE (MISSING_THE_POINT):**
Your opening must follow this structure:
1. First sentence: State what conventional coverage focuses on
2. Second sentence: Pivot with "But" or "Yet" or "The real story is..."
3. Rest of paragraph: Explain the insight everyone is missing

TEMPLATE: "Everyone is watching [X]. The real story is [Y]: [explanation of why Y matters more]."

EXAMPLE: "Everyone's parsing tariff percentages. The real story is supply chain realignment: European manufacturers are moving production to Morocco and Vietnam, not back to Detroit."
"""
    
    elif "UNINTENDED_CONSEQUENCE" in angle_upper or "UNINTENDED CONSEQUENCE" in angle_upper:
        return """
**HOW TO EXECUTE (UNINTENDED_CONSEQUENCE):**
Your opening must show the irony/contradiction:
1. State the policy's intended goal
2. Show what it's actually causing (the opposite or something unexpected)
3. Explain the mechanism of this backfire

TEMPLATE: "[Policy/Action] was designed to achieve [X]. Instead, it's driving [Y], because [mechanism]."

EXAMPLE: "Sanctions designed to isolate Moscow are accelerating de-dollarization. By freezing Russian reserves, Washington showed every central bank that dollar assets are conditional—triggering a quiet scramble into gold and yuan."
"""
    
    elif "CONSTRAINT_REVEALED" in angle_upper or "CONSTRAINT REVEALED" in angle_upper:
        return """
**HOW TO EXECUTE (CONSTRAINT_REVEALED):**
Your opening must expose the structural limit:
1. Describe the event/crisis
2. Show how it exposes a deeper constraint that was always there
3. Explain why this constraint will persist

TEMPLATE: "[Event] exposed what was always true: [Actor] cannot [do X] because [structural constraint]."

EXAMPLE: "Germany's energy crisis exposed what was always true: Europe cannot sustain manufacturing without cheap Russian gas. Geography and pipeline physics haven't changed—only the willingness to admit it."
"""
    
    elif "TRAJECTORY_SHIFT" in angle_upper or "TRAJECTORY SHIFT" in angle_upper:
        return """
**HOW TO EXECUTE (TRAJECTORY_SHIFT):**
Your opening must mark the inflection point:
1. State what the previous pattern/policy was
2. Identify the specific event/decision that changed direction
3. Explain what the new trajectory is

TEMPLATE: "After [time period] of [old pattern], [Actor] has pivoted to [new approach]. The shift: [what specifically changed]."

EXAMPLE: "After years of hedging between Washington and Beijing, Ankara has picked a side. The shift: Turkey just signed a $20B defense procurement deal with China, ending decades of NATO equipment exclusivity."
"""
    
    elif "MIND_CLEARING" in angle_upper or "MIND CLEARING" in angle_upper:
        return """
**HOW TO EXECUTE (MIND_CLEARING):**
Your opening must cut through complexity:
1. Acknowledge the confusing surface details
2. Identify the ONE variable that actually drives everything
3. Show how understanding this variable clarifies the situation

TEMPLATE: "Forget [all the noise]. This is about [the one thing that matters]: [explanation]."

EXAMPLE: "Forget the rhetoric about democracy and human rights. This is about port access: whoever controls Djibouti controls the chokepoint for 30% of global oil shipments."
"""
    
    else:
        # Fallback for unknown angle type or custom angles
        return """
**HOW TO EXECUTE THIS ANGLE:**
Your opening paragraph must deliver the unique insight specified in the angle above.
Don't just describe what happened—show readers what makes this story intellectually interesting.
Lead with the tension, contradiction, or surprise that justifies their attention.
"""


def _build_writer_prompt(input_data: WriterInput) -> str:
    """Build the user prompt for the writer agent."""

    # Convert analyst output to dict for the prompt
    analyst_data = input_data.analyst_output.model_dump()

    # Build events summary for linking
    events_summary = []
    for event in input_data.events_for_linking[:20]:
        events_summary.append(
            {
                "id": event.get("id", ""),
                "title": event.get("title", "")[:100],
                "severity": event.get("severity", 5),
            }
        )

    # Use blueprint if provided, otherwise fall back to default structure
    section_structure = _get_section_structure(input_data.section_type, input_data.section_blueprint)
    
    # Note if we're using a blueprint
    blueprint_note = ""
    if input_data.section_blueprint:
        blueprint_note = """
## 📋 STRUCTURE AGENT BLUEPRINT

You have been given a detailed paragraph-by-paragraph plan from the Structure Agent.
FOLLOW IT EXACTLY. Each paragraph has a purpose, word target, and events to cite.
The hook and closing drafts are starting points - you may refine them but keep their intent.
"""

    revision_section = ""
    if input_data.revision_feedback:
        previous_draft_block = ""
        if input_data.previous_draft:
            previous_draft_block = f"""
### YOUR PREVIOUS DRAFT (that failed):

```markdown
{input_data.previous_draft}
```

"""
        revision_section = f"""
## ⚠️ REVISION REQUIRED

Your previous draft FAILED the Content Critic audit. 

{previous_draft_block}
### CRITIC FEEDBACK:

{input_data.revision_feedback}

### YOUR TASK:

1. Read your previous draft above
2. Understand each issue the Critic identified
3. Rewrite the ENTIRE draft, fixing ALL issues
4. Use the Analyst output below as your source of truth for facts

Do NOT just make minor edits. Rewrite with the feedback in mind.
"""

    # Format editorial angle if provided (for featured stories)
    angle_section = ""
    if input_data.editorial_angle:
        # Determine angle type and provide specific execution instructions
        angle_instructions = _get_angle_instructions(input_data.editorial_angle)
        
        angle_section = f"""
## 🎯 EDITORIAL ANGLE (Your "Nub")

The Architect wants this story to emphasize:
**{input_data.editorial_angle}**

{angle_instructions}

**CRITICAL:** Your opening paragraph MUST deliver this angle. Don't just describe events—execute the angle.
"""

    # Format sacred elements if provided
    sacred_section = ""
    if input_data.sacred_elements:
        sacred_parts = []
        if input_data.sacred_elements.proper_nouns:
            sacred_parts.append(f"**Names (use exactly):** {', '.join(input_data.sacred_elements.proper_nouns[:10])}")
        if input_data.sacred_elements.statistics:
            sacred_parts.append(f"**Numbers (use exactly):** {', '.join(input_data.sacred_elements.statistics[:10])}")
        if input_data.sacred_elements.dates:
            sacred_parts.append(f"**Dates (use exactly):** {', '.join(input_data.sacred_elements.dates[:10])}")
        if input_data.sacred_elements.quotes:
            sacred_parts.append(f"**Quotes (word-for-word):** {json.dumps(input_data.sacred_elements.quotes[:3])}")
        
        if sacred_parts:
            sacred_section = f"""
## 🔒 SACRED ELEMENTS (DO NOT ALTER)

These facts are extracted from source documents. Use them EXACTLY as written.
Do NOT paraphrase, round numbers, or change names.

{chr(10).join(sacred_parts)}
"""

    prompt = f"""
## TASK

Write a {input_data.section_type.upper()} section for the weekly intelligence briefing.
Target word count: {input_data.word_limit} words.

{revision_section}
{angle_section}
{blueprint_note}
{sacred_section}

## STRUCTURED ANALYSIS (Your input - this is already done, just communicate it)

```json
{json.dumps(analyst_data, indent=2)}
```

NOTE: If external_sources is not empty, the Analyst used web search for context.
You can cite these sources using [1], [2] style and list them at the section end.

## AVAILABLE EVENTS FOR LINKING

These events from the database can be cited inline. 
Extract SHORT PHRASES (1-4 words) from the titles to use as link text.

GOOD EXAMPLES:
- Title: "US President Orders Venezuela Strike; Maduro Kidnapped"
  → "Maduro was [reportedly kidnapped](event_id) by US forces."
  
- Title: "Trump Threatens 10% Tariffs on European Nations Over Greenland"  
  → "Trump's [tariff threats](event_id) shocked allies."
  
- Title: "Iran Protest Death Toll Exceeds 5,000 Amid Crackdown"
  → "The [brutal crackdown](event_id) killed over 5,000."

BAD EXAMPLES (too long):
❌ "The [US President ordered a Venezuela strike and Maduro was kidnapped](event_id)..."
❌ "Trump's [threat of 10% tariffs on eight European nations over Greenland](event_id)..."

```json
{json.dumps(events_summary, indent=2)}
```

## OUTPUT STRUCTURE

{section_structure}

Write the section now. Remember:
- The Editorial Angle given to you
- Lead with Key Judgment (constraint + probability + implication)
- Entity-dense prose (names, numbers, dates)
- Sherman Kent probability terms
- NO banned phrases
- Link to events using https://realpolitik.world/?event=EVENT_ID format
- Preserve all SACRED ELEMENTS exactly as given
"""

    return prompt


@with_retry(max_attempts=3, initial_delay=2.0, max_delay=30.0)
async def run_writer_agent(
    input_data: WriterInput,
    client: genai.Client | None = None,
) -> str:
    """
    Run the Writer Agent to produce polished prose.

    Args:
        input_data: The input containing analyst output and context
        client: Optional Gemini client (created if not provided)

    Returns:
        Markdown string ready for publication
    """
    config = get_config()

    if client is None:
        client = genai.Client(api_key=config.gemini_api_key)

    model = config.models.writer
    region = input_data.analyst_output.region
    logger.info(f"   ✍️ [{region}] Running Writer Agent with {model}...")

    prompt = _build_writer_prompt(input_data)

    # Configure for high-quality prose - only set params if explicitly configured
    # Writer doesn't need search tools - it uses pre-analyzed events from Analyst
    gen_config = types.GenerateContentConfig(
        system_instruction=WRITER_SYSTEM_PROMPT,
    )
    
    # Only set temperature if explicitly configured
    if config.writer_temperature is not None:
        gen_config.temperature = config.writer_temperature
    
    # Only set max_output_tokens if explicitly configured
    if config.writer_max_output_tokens is not None:
        gen_config.max_output_tokens = config.writer_max_output_tokens
    
    # Only set thinking level if explicitly configured
    if config.writer_thinking_level is not None:
        gen_config.thinking_config = types.ThinkingConfig(
            thinking_level=config.writer_thinking_level,
        )

    # Log config for debugging
    log_model_config("Writer", model, gen_config)

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=gen_config,
    )

    content = response.text.strip()
    
    # Strip any leaked metadata labels (defensive)
    content = strip_metadata_labels(content, region=region)

    # Count words for logging
    initial_word_count = len(content.split())
    
    # Apply Chain of Density if enabled (works in all modes now)
    if input_data.apply_chain_of_density:
        logger.info(f"   🔄 [{region}] Applying Chain of Density (3 iterations max)...")
        try:
            content = await _apply_chain_of_density(
                draft=content,
                client=client,
                model=model,
                iterations=3,
                sacred_elements=input_data.sacred_elements,
            )
            final_word_count = len(content.split())
            logger.info(f"   ✅ [{region}] CoD complete: {initial_word_count} → {final_word_count} words")
        except Exception as e:
            logger.warning(f"   ⚠️ [{region}] CoD failed, using initial draft: {e}")
    
    # Clean up malformed event links (IDs without descriptive text)
    import re
    
    # Pattern: [EVENT_ID] where EVENT_ID is a hex string
    # Replace with proper link format using a generic description
    def fix_bare_id(match):
        event_id = match.group(1)
        return f"[related event](https://realpolitik.world/?event={event_id})"
    
    content = re.sub(
        r'\[([a-f0-9]{16})\](?!\()',  # [hexid] NOT followed by (url)
        fix_bare_id,
        content
    )
    
    # Remove erroneous code fences that Writer sometimes adds
    # Pattern: ```text\n...content...\n``` or ```\n...content...\n```
    if content.startswith('```'):
        # Strip opening fence
        lines = content.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]  # Remove first line
        # Strip closing fence
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        content = '\n'.join(lines)
    
    word_count = len(content.split())
    logger.info(f"   ✅ [{region}] Writer complete: {word_count} words")

    return content


# =============================================================================
# SYNC WRAPPER
# =============================================================================


def run_writer_agent_sync(
    input_data: WriterInput,
    client: genai.Client | None = None,
) -> str:
    """Synchronous wrapper for run_writer_agent."""
    import asyncio

    return asyncio.run(run_writer_agent(input_data, client))
