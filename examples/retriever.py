"""
Dynamic few-shot example retrieval for the Stylist agent.

Retrieves relevant transformation pairs based on:
- Topic (Elections, Conflict, Economic, Diplomatic)
- Tone (Somber, Wry, Urgent, Analytical)
- Semantic similarity (optional, for future enhancement)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .economist_style import ECONOMIST_EXAMPLES, STRATFOR_EXAMPLES


@dataclass
class StyleExample:
    """A curated style transformation example."""
    
    id: str
    topic: str
    tone: str
    before: str
    after: str
    notes: str
    
    def to_prompt_format(self, include_notes: bool = False) -> str:
        """Format for inclusion in a prompt."""
        result = f"""
### Example Transformation

**Before (flat prose):**
{self.before}

**After (premium prose):**
{self.after}
""".strip()
        
        if include_notes:
            result += f"\n\n**What makes it work:**\n{self.notes}"
        
        return result


class ExampleRetriever:
    """
    Retrieves relevant style examples for few-shot learning.
    
    Usage:
        retriever = ExampleRetriever(preset="economist")
        examples = retriever.get_examples(topic="Conflict", tone="Urgent", limit=2)
        prompt_text = retriever.format_for_prompt(examples)
    """
    
    def __init__(self, preset: str = "mixed"):
        """
        Initialize with curated examples.
        
        Args:
            preset: Style preset to use - "economist", "stratfor", or "mixed" (default)
        """
        self._examples: list[StyleExample] = []
        
        # Load examples based on preset
        if preset in ("economist", "mixed"):
            for ex in ECONOMIST_EXAMPLES:
                self._examples.append(StyleExample(
                    id=ex["id"],
                    topic=ex["topic"],
                    tone=ex["tone"],
                    before=ex["before"],
                    after=ex["after"],
                    notes=ex["notes"],
                ))
        
        if preset in ("stratfor", "mixed"):
            for ex in STRATFOR_EXAMPLES:
                self._examples.append(StyleExample(
                    id=ex["id"],
                    topic=ex["topic"],
                    tone=ex["tone"],
                    before=ex["before"],
                    after=ex["after"],
                    notes=ex["notes"],
                ))
    
    def get_examples(
        self,
        topic: Optional[str] = None,
        tone: Optional[str] = None,
        limit: int = 3,
        include_diverse: bool = True,
    ) -> list[StyleExample]:
        """
        Retrieve examples matching the given criteria.
        
        Args:
            topic: Filter by topic (Elections, Conflict, Economic, Diplomatic)
            tone: Filter by tone (Somber, Wry, Urgent, Analytical)
            limit: Maximum number of examples to return
            include_diverse: If True, ensure variety in returned examples
            
        Returns:
            List of matching StyleExample objects
        """
        candidates = self._examples.copy()
        
        # Score each example based on match
        scored = []
        for ex in candidates:
            score = 0
            if topic and ex.topic.lower() == topic.lower():
                score += 2
            if tone and ex.tone.lower() == tone.lower():
                score += 2
            # Give base score to ensure we return something
            score += 1
            scored.append((score, ex))
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        if include_diverse and limit > 1:
            # Ensure we don't return all same topic/tone
            result = []
            used_topics = set()
            used_tones = set()
            
            for score, ex in scored:
                # First, prioritize high-scoring matches
                if len(result) == 0 or score == scored[0][0]:
                    result.append(ex)
                    used_topics.add(ex.topic)
                    used_tones.add(ex.tone)
                # Then, add diverse examples
                elif len(result) < limit:
                    if ex.topic not in used_topics or ex.tone not in used_tones:
                        result.append(ex)
                        used_topics.add(ex.topic)
                        used_tones.add(ex.tone)
                
                if len(result) >= limit:
                    break
            
            # Fill remaining slots if needed
            if len(result) < limit:
                for score, ex in scored:
                    if ex not in result:
                        result.append(ex)
                    if len(result) >= limit:
                        break
            
            return result[:limit]
        else:
            return [ex for _, ex in scored[:limit]]
    
    def format_for_prompt(
        self,
        examples: list[StyleExample],
        include_notes: bool = False,
    ) -> str:
        """
        Format examples for inclusion in a prompt.
        
        Args:
            examples: List of StyleExample objects
            include_notes: Whether to include transformation notes
            
        Returns:
            Formatted string for prompt injection
        """
        if not examples:
            return ""
        
        sections = ["## Style Reference Examples\n"]
        sections.append("Study these transformations. They demonstrate the voice we want.\n")
        
        for i, ex in enumerate(examples, 1):
            sections.append(f"### Example {i} ({ex.topic}, {ex.tone} tone)\n")
            sections.append(ex.to_prompt_format(include_notes=include_notes))
            sections.append("")
        
        return "\n".join(sections)
    
    def get_all_topics(self) -> list[str]:
        """Return all unique topics."""
        return list(set(ex.topic for ex in self._examples))
    
    def get_all_tones(self) -> list[str]:
        """Return all unique tones."""
        return list(set(ex.tone for ex in self._examples))


# Convenience function for quick access
def get_style_examples(
    topic: Optional[str] = None,
    tone: Optional[str] = None,
    limit: int = 3,
    preset: str = "mixed",
) -> str:
    """
    Get formatted style examples for a prompt.
    
    Args:
        topic: Filter by topic
        tone: Filter by tone
        limit: Max examples
        preset: Style preset - "economist", "stratfor", or "mixed" (default)
        
    Returns:
        Formatted string ready for prompt injection
    """
    retriever = ExampleRetriever(preset=preset)
    examples = retriever.get_examples(topic=topic, tone=tone, limit=limit)
    return retriever.format_for_prompt(examples, include_notes=True)
