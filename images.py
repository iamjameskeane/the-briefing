"""
The Briefing - Image Generation using Nano Banana (Gemini Native Image Generation)

Parses IMAGE_PROMPT blocks from The Briefing markdown and generates images
using gemini-2.5-flash-image model.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

from utils import with_retry, logger
from config import get_config


@dataclass
class ImagePrompt:
    """Parsed image prompt from markdown."""
    location: str
    prompt: str
    style: str
    mood: str
    
    def get_aspect_ratio(self) -> str:
        """Determine aspect ratio based on location."""
        location_lower = self.location.lower()
        if "header" in location_lower or "hero" in location_lower:
            return "16:9"
        elif "inline" in location_lower or "thumbnail" in location_lower:
            return "4:3"
        elif "icon" in location_lower or "square" in location_lower:
            return "1:1"
        else:
            return "16:9"  # Default to wide format


def get_genai_client() -> genai.Client:
    """Configure and return Gemini client."""
    config = get_config()
    if not config.gemini_api_key:
        raise ValueError("GEMINI_API_KEY not set")
    return genai.Client(api_key=config.gemini_api_key)


def parse_image_prompts(markdown: str) -> list[ImagePrompt]:
    """
    Extract IMAGE_PROMPT blocks from The Briefing markdown.
    
    Supports two formats:
    
    1. Structured format:
    <!-- IMAGE_PROMPT
    location: Header image
    prompt: Editorial illustration of...
    style: editorial illustration
    mood: tense
    -->
    
    2. Simple inline format (from Writer Agent):
    <!-- IMAGE_PROMPT: A high-contrast visualization of... -->
    
    Args:
        markdown: The Briefing markdown content.
    
    Returns:
        List of ImagePrompt objects.
    """
    prompts = []
    
    # Pattern 1: Structured multi-line format
    pattern1 = r'<!-- IMAGE_PROMPT\n(.*?)-->'
    matches = re.findall(pattern1, markdown, re.DOTALL)
    
    for match in matches:
        prompt_data = {
            "location": "",
            "prompt": "",
            "style": "editorial illustration",
            "mood": "analytical",
        }
        
        for line in match.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if key in prompt_data:
                    prompt_data[key] = value
        
        if prompt_data["prompt"]:  # Only add if we have a prompt
            prompts.append(ImagePrompt(
                location=prompt_data["location"] or "Inline",
                prompt=prompt_data["prompt"],
                style=prompt_data["style"],
                mood=prompt_data["mood"],
            ))
    
    # Pattern 2: Simple inline format <!-- IMAGE_PROMPT: prompt text -->
    pattern2 = r'<!-- IMAGE_PROMPT:\s*(.+?)\s*-->'
    inline_matches = re.findall(pattern2, markdown, re.DOTALL)
    
    for i, match in enumerate(inline_matches):
        # Determine location based on position in document
        if i == 0:
            location = "Header"
        elif "featured" in markdown[:markdown.find(match)].lower():
            location = "Featured Analysis"
        else:
            location = f"Section {i}"
        
        prompts.append(ImagePrompt(
            location=location,
            prompt=match.strip(),
            style="war room",
            mood="tactical",
        ))
    
    return prompts


@with_retry(max_attempts=3, initial_delay=2.0, max_delay=30.0)
def _generate_image_with_retry(
    client: genai.Client,
    prompt: str,
    aspect_ratio: str,
) -> bytes | None:
    """Generate image with retry logic."""
    config = get_config()
    response = client.models.generate_content(
        model=config.models.image,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_modalities=['IMAGE'],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
            )
        )
    )
    
    # Extract image data from response
    for part in response.candidates[0].content.parts:
        if hasattr(part, 'inline_data') and part.inline_data is not None:
            return part.inline_data.data
    
    return None


def enhance_prompt_with_style(prompt: str) -> str:
    """
    Enhance prompt with realpolitik.world professional editorial visual identity.
    Ensures consistent style and removes any date references.
    
    Based on Nano Banana prompt guide for The Briefing newsletter visuals.
    """
    from datetime import datetime, timezone
    current_year = datetime.now(timezone.utc).year
    
    # Remove any old year references that might be in the prompt
    enhanced = prompt
    for year in range(2020, current_year):
        enhanced = enhanced.replace(str(year), str(current_year))
    
    # Ensure key Briefing style elements are present
    required_elements = []
    
    if "no text" not in enhanced.lower():
        required_elements.append("no text")
    if "no faces" not in enhanced.lower():
        required_elements.append("no faces")
    if "professional" not in enhanced.lower() and "editorial" not in enhanced.lower():
        required_elements.append("professional news magazine style")
    if "muted" not in enhanced.lower() and "color palette" not in enhanced.lower():
        required_elements.append("muted color palette")
    if "lighting" not in enhanced.lower():
        required_elements.append("dramatic lighting")
        
    if required_elements:
        enhanced += ", " + ", ".join(required_elements)
    
    return enhanced


def generate_image(prompt: str, aspect_ratio: str = "16:9") -> bytes | None:
    """
    Generate an image using Nano Banana (gemini-2.5-flash-image).
    
    Args:
        prompt: The image generation prompt.
        aspect_ratio: Aspect ratio (16:9, 4:3, 1:1, etc.)
    
    Returns:
        Image bytes (PNG format) or None if generation fails.
    """
    client = get_genai_client()
    
    # Enhance prompt with War Room style
    enhanced_prompt = enhance_prompt_with_style(prompt)
    logger.info(f"   Enhanced prompt: {enhanced_prompt[:100]}...")
    
    try:
        return _generate_image_with_retry(client, enhanced_prompt, aspect_ratio)
    except Exception as e:
        logger.warning(f"Image generation failed: {type(e).__name__}: {e}")
        return None


def generate_briefing_images(
    markdown: str,
    output_dir: str | Path,
) -> list[tuple[str, str]]:
    """
    Generate all images for The Briefing newsletter.
    
    Args:
        markdown: The Briefing markdown content.
        output_dir: Directory to save generated images.
    
    Returns:
        List of tuples (location, filepath) for generated images.
    """
    from config import get_config
    config = get_config()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    prompts = parse_image_prompts(markdown)
    
    if not prompts:
        print("   📷 No IMAGE_PROMPT blocks found in markdown")
        return []
    
    # Apply max images limit from config
    max_images = config.max_images_per_briefing
    if len(prompts) > max_images:
        print(f"\n   📷 Found {len(prompts)} image prompts, limiting to {max_images}")
        prompts = prompts[:max_images]
    else:
        print(f"\n   📷 Found {len(prompts)} image prompts to generate")
    
    generated = []
    
    for i, image_prompt in enumerate(prompts):
        print(f"   🎨 Generating image {i + 1}/{len(prompts)}: {image_prompt.location}")
        
        aspect_ratio = image_prompt.get_aspect_ratio()
        image_data = generate_image(image_prompt.prompt, aspect_ratio)
        
        if image_data:
            # Create filename from location
            safe_location = re.sub(r'[^a-z0-9]+', '_', image_prompt.location.lower())
            filename = f"briefing_{safe_location}_{i}.png"
            filepath = output_path / filename
            
            # Save image
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            generated.append((image_prompt.location, str(filepath)))
            print(f"      ✅ Saved: {filepath}")
        else:
            print(f"      ❌ Failed to generate image for: {image_prompt.location}")
    
    return generated


def remove_image_prompt_blocks(markdown: str) -> str:
    """
    Remove IMAGE_PROMPT comment blocks from markdown.
    
    Call this after generating images and inserting them.
    
    Args:
        markdown: The Briefing markdown content.
    
    Returns:
        Markdown with IMAGE_PROMPT blocks removed.
    """
    pattern = r'<!-- IMAGE_PROMPT\n.*?-->\n?'
    return re.sub(pattern, '', markdown, flags=re.DOTALL)


def insert_images_into_markdown(
    markdown: str,
    generated_images: list[tuple[str, str]],
    base_url: str = "https://your-bucket.r2.dev/briefing/images/",
) -> str:
    """
    Insert generated images into markdown at appropriate locations.
    
    This is a helper that creates markdown image tags before the
    IMAGE_PROMPT blocks, then removes the blocks.
    
    Args:
        markdown: The Briefing markdown content.
        generated_images: List of (location, filepath) tuples.
        base_url: Base URL for image hosting.
    
    Returns:
        Updated markdown with images inserted.
    """
    result = markdown
    
    for location, filepath in generated_images:
        filename = Path(filepath).name
        image_url = f"{base_url}{filename}"
        
        # Find the IMAGE_PROMPT block for this location
        pattern = rf'(<!-- IMAGE_PROMPT\nlocation: {re.escape(location)}.*?-->)'
        
        # Insert image markdown before the block
        image_md = f"![{location}]({image_url})\n\n"
        result = re.sub(pattern, image_md + r'\1', result, flags=re.DOTALL)
    
    # Remove all IMAGE_PROMPT blocks
    result = remove_image_prompt_blocks(result)
    
    return result


if __name__ == "__main__":
    """Test image generation standalone."""
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Test with a sample prompt
    test_prompt = """Editorial illustration of naval vessels in foggy strait waters, 
    aircraft carrier silhouettes, two coastlines in tension, muted blue 
    and gray color palette, dramatic lighting, professional news magazine 
    style, no text, 16:9 aspect ratio"""
    
    print("🎨 Testing Nano Banana image generation...")
    
    try:
        image_data = generate_image(test_prompt, "16:9")
        
        if image_data:
            output_path = Path(__file__).parent / "test_image.png"
            with open(output_path, 'wb') as f:
                f.write(image_data)
            print(f"✅ Image saved to: {output_path}")
            print(f"   Size: {len(image_data)} bytes")
        else:
            print("❌ No image data returned")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
