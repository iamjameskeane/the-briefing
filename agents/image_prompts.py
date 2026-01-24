"""
Image Prompt Enhancement Agent.

Takes Writer's image intent and transforms it into a high-quality
Nano Banana (gemini-2.5-flash-image) prompt.
"""

import logging
from google import genai
from google.genai import types

from config import get_config
from utils import log_model_config

logger = logging.getLogger("briefing")

# =============================================================================
# SYSTEM PROMPT FOR IMAGE ENHANCEMENT
# =============================================================================

IMAGE_ENHANCER_PROMPT = """You are an expert at crafting image generation prompts for a professional geopolitical intelligence newsletter.

Your task: Transform a simple image concept into a detailed, evocative prompt for Gemini's image generation model.

## YOUR CREATIVE MANDATE

Each image should feel DISTINCT. Vary your approach across these dimensions:
- **Art style**: Choose ONE per image - photojournalistic, oil painting, woodcut print, lithograph, data visualization, satellite imagery, architectural rendering, infrared photography, blueprint schematic
- **Color approach**: Don't default to the same palette. Consider: monochrome, duotone, high contrast, desaturated, warm-dominant, cool-dominant, complementary colors, analogous colors
- **Perspective**: Vary between aerial/satellite view, ground-level, close-up detail, wide establishing shot, isometric, cross-section
- **Lighting**: golden hour, harsh midday, twilight, moonlit, artificial/industrial, backlit silhouettes, chiaroscuro

## QUALITY ANCHORS (always include)
- Professional magazine/editorial quality
- No text, labels, or annotations
- No recognizable faces (use silhouettes, backs of heads, distant figures)
- No specific national flags (use abstract geography, terrain)
- No explicit violence (use aftermath, tension, symbolic imagery)
- Specify aspect ratio (16:9 for headers, 4:3 for sections)

## SUBJECT TRANSLATION (optional inspiration)
- Tensions → standoff compositions, pressure atmospheres, divided landscapes
- Conflict → aftermath landscapes, fractured terrain, distant smoke
- Economic → trade routes as light trails, infrastructure networks, port activity
- Military → vessel silhouettes, strategic overlays, defense perimeters
- Diplomacy → summit compositions, negotiation atmospheres, handshake shadows

## OUTPUT FORMAT

Produce a single paragraph prompt (60-100 words). Be specific and evocative.
Start directly with the visual description - no prefixes like "IMAGE:" or "Prompt:".

Example variety:
- "Woodcut-style illustration of Arctic shipping lanes..."
- "Satellite perspective photograph of Middle Eastern oil infrastructure..."
- "Duotone lithograph depicting European energy grid..."
- "High-contrast photojournalistic image of..."
"""


# =============================================================================
# IMAGE INTENT PATTERNS
# =============================================================================

def extract_image_intents(markdown: str) -> list[tuple[int, str, str]]:
    """
    Extract image intents from the markdown.
    
    Returns list of (position, intent, context) tuples.
    The Writer Agent should output these in format:
    <!-- IMAGE: brief description of desired visual -->
    
    Or the older format:
    <!-- IMAGE_PROMPT: ... -->
    """
    import re
    
    intents = []
    
    # Pattern 1: New simple intent format
    pattern1 = re.compile(
        r'<!-- IMAGE:\s*(.+?)\s*-->',
        re.IGNORECASE | re.DOTALL
    )
    
    # Pattern 2: Old IMAGE_PROMPT format (fallback)
    pattern2 = re.compile(
        r'<!-- IMAGE_PROMPT:\s*(.+?)\s*-->',
        re.IGNORECASE | re.DOTALL
    )
    
    for pattern in [pattern1, pattern2]:
        for match in pattern.finditer(markdown):
            # Get surrounding context (800 chars before and 1200 chars after)
            # This captures the section headline AND the first few paragraphs
            start = max(0, match.start() - 800)
            end = min(len(markdown), match.end() + 1200)
            context = markdown[start:end]
            
            intents.append((
                match.start(),
                match.group(1).strip(),
                context
            ))
    
    return intents


async def enhance_image_prompt(
    intent: str,
    context: str,
    region: str | None = None,
) -> str:
    """
    Transform a simple image intent into a high-quality Nano Banana prompt.
    
    Args:
        intent: The Writer's simple image description
        context: Surrounding text for additional context
        region: Optional region name for additional context
        
    Returns:
        Enhanced prompt suitable for Nano Banana
    """
    config = get_config()
    client = genai.Client()
    
    # Build the enhancement request
    user_prompt = f"""Transform this image concept into a detailed image generation prompt.

## IMAGE CONCEPT
{intent}

## SECTION CONTENT (for context)
{context[:1500]}

## REGION (if relevant)
{region or "Global"}

Generate the enhanced prompt now:"""

    gen_config = types.GenerateContentConfig(
        system_instruction=IMAGE_ENHANCER_PROMPT,
    )
    
    # Only set temperature if explicitly configured
    if config.image_temperature is not None:
        gen_config.temperature = config.image_temperature
    
    # Only set max_output_tokens if explicitly configured
    if config.image_max_output_tokens is not None:
        gen_config.max_output_tokens = config.image_max_output_tokens
    
    # Log config for debugging
    log_model_config("ImageEnhancer", config.models.theme, gen_config)
    
    response = await client.aio.models.generate_content(
        model=config.models.theme,  # Use fast model for prompt enhancement
        contents=user_prompt,
        config=gen_config,
    )
    
    enhanced = response.text.strip()
    
    # Ensure key style elements are present
    if "no text" not in enhanced.lower():
        enhanced += ", no text"
    if "no faces" not in enhanced.lower():
        enhanced += ", no faces"
    if "aspect ratio" not in enhanced.lower():
        enhanced += ", 16:9 aspect ratio"
    
    return enhanced


async def enhance_all_image_prompts(
    markdown: str,
    region_name: str | None = None,
) -> tuple[str, list[str]]:
    """
    Find all image intents in markdown and enhance them.
    
    Returns:
        Tuple of (updated_markdown, list_of_enhanced_prompts)
    """
    import re
    
    intents = extract_image_intents(markdown)
    
    if not intents:
        logger.info("   📷 No image intents found in markdown")
        return markdown, []
    
    logger.info(f"   📷 Found {len(intents)} image intents, enhancing...")
    
    enhanced_prompts = []
    updated_markdown = markdown
    
    # Process in reverse order to maintain positions
    for position, intent, context in sorted(intents, reverse=True, key=lambda x: x[0]):
        try:
            enhanced = await enhance_image_prompt(intent, context, region_name)
            enhanced_prompts.append(enhanced)
            
            # Replace the original marker with the enhanced prompt
            # Find the original marker at this position
            pattern = re.compile(
                r'<!-- IMAGE(?:_PROMPT)?:\s*.+?\s*-->',
                re.IGNORECASE | re.DOTALL
            )
            
            # Find the match at approximately this position
            for match in pattern.finditer(updated_markdown):
                if abs(match.start() - position) < 50:  # Allow some tolerance
                    updated_markdown = (
                        updated_markdown[:match.start()] + 
                        f"<!-- IMAGE_PROMPT: {enhanced} -->" +
                        updated_markdown[match.end():]
                    )
                    break
                    
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to enhance image prompt: {e}")
            # Keep original on failure
            enhanced_prompts.append(intent)
    
    return updated_markdown, enhanced_prompts


# =============================================================================
# STANDALONE IMAGE PROMPT GENERATOR
# =============================================================================

async def generate_header_image_prompt(
    title: str,
    summary: str,
    archetype: str,
) -> str:
    """
    Generate a header image prompt for the briefing based on the featured analysis.
    
    Args:
        title: The featured analysis title
        summary: Brief summary of the key judgment
        archetype: The geopolitical archetype (e.g., "Security Dilemma")
        
    Returns:
        Enhanced prompt for the header image
    """
    intent = f"Header image for: {title}. Theme: {archetype}. Key tension: {summary[:100]}"
    context = f"This is the header image for a weekly intelligence briefing. Featured topic: {title}"
    
    return await enhance_image_prompt(intent, context, region=None)


async def generate_section_image_prompt(
    region: str,
    archetype: str,
    key_judgment: str,
) -> str:
    """
    Generate an image prompt for a regional section.
    
    Args:
        region: The region name
        archetype: The geopolitical archetype
        key_judgment: The key judgment for this region
        
    Returns:
        Enhanced prompt for the section image
    """
    intent = f"{region} section: {archetype}. Shows: {key_judgment[:80]}"
    context = f"Regional briefing section for {region}, analyzing {archetype} dynamics"
    
    return await enhance_image_prompt(intent, context, region=region)
