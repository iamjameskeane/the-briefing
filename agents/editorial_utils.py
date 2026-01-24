"""
Editorial Decision Frameworks

Shared utilities for editorial decision-making (PIC scoring, Delta Test, etc.)
Used by both Editor agent and tests.
"""

from __future__ import annotations

from state import PICScore
from .schemas import AnalystOutput


# =============================================================================
# SIGNAL CORROBORATION SCORING
# =============================================================================

def calculate_signal_corroboration_score(
    events: list[dict],
    source_diversity: dict[str, int] | None = None,
    region: str = "",
) -> int:
    """
    Calculate Signal Corroboration Score (SCS) for event validation.
    
    From research: An event is only elevated if it meets minimum citation
    threshold from independent source clusters.
    
    Scoring:
    - Single source: 1 (noise tier)
    - 2-3 independent sources: 2-3 (monitoring tier)
    - Multiple authoritative + quantitative: 4-5 (candidate/critical tier)
    
    Returns:
        SCS score 1-5
    """
    if not events:
        return 1
    
    # Use provided source diversity if available
    if source_diversity and region in source_diversity:
        distinct_sources = source_diversity[region]
    else:
        # Estimate from events
        sources = set()
        for event in events:
            source = event.get("source", "unknown")
            sources.add(source.lower() if source else "unknown")
        distinct_sources = len(sources)
    
    # Check for quantitative confirmation
    has_quantitative = any(
        event.get("severity", 0) >= 7 or
        any(char.isdigit() for char in str(event.get("title", "")))
        for event in events
    )
    
    # Check for official statements
    has_official = any(
        "official" in str(event.get("summary", "")).lower() or
        "government" in str(event.get("source", "")).lower() or
        "ministry" in str(event.get("source", "")).lower()
        for event in events
    )
    
    # Calculate SCS
    if distinct_sources >= 5 and has_quantitative and has_official:
        return 5  # Critical tier
    elif distinct_sources >= 3 and (has_quantitative or has_official):
        return 4  # Candidate tier
    elif distinct_sources >= 2:
        return 3  # Monitoring tier
    elif distinct_sources == 1:
        return 2  # Single source
    else:
        return 1  # Noise tier


# =============================================================================
# THEMATIC SYNCHRONIZATION ANALYSIS
# =============================================================================

def analyze_thematic_synchronization(
    cross_regional_connections: list[str],
    num_regions: int,
) -> dict:
    """
    Analyze cross-regional connections to detect thematic synchronization.
    
    Returns:
        Dict with:
        - synchronization_pct: Percentage of regions sharing a common driver
        - dominant_theme: The most common theme if any
        - recommendation: "THEMATIC", "REGIONAL", or "HYBRID"
    """
    if not cross_regional_connections or num_regions < 2:
        return {
            "synchronization_pct": 0,
            "dominant_theme": None,
            "recommendation": "REGIONAL",
            "explanation": "No cross-regional connections identified"
        }
    
    # Count theme occurrences (simple keyword extraction)
    theme_keywords = {}
    for conn in cross_regional_connections:
        # Extract potential theme keywords (simplified)
        conn_lower = conn.lower()
        
        # Check for common thematic keywords
        potential_themes = [
            ("US", ["us ", "american", "washington", "biden", "trump", "united states"]),
            ("China", ["china", "chinese", "beijing", "xi jinping"]),
            ("Russia", ["russia", "russian", "moscow", "putin", "kremlin"]),
            ("Energy", ["oil", "gas", "energy", "opec", "pipeline"]),
            ("Trade", ["trade", "tariff", "export", "import", "sanctions"]),
            ("Security", ["military", "defense", "nato", "war", "conflict"]),
            ("Economic", ["economy", "inflation", "debt", "gdp", "recession"]),
        ]
        
        for theme, keywords in potential_themes:
            if any(kw in conn_lower for kw in keywords):
                theme_keywords[theme] = theme_keywords.get(theme, 0) + 1
    
    if not theme_keywords:
        return {
            "synchronization_pct": 0,
            "dominant_theme": None,
            "recommendation": "REGIONAL",
            "explanation": "No dominant theme detected in connections"
        }
    
    # Find dominant theme
    dominant_theme = max(theme_keywords, key=theme_keywords.get)
    dominant_count = theme_keywords[dominant_theme]
    
    # Calculate synchronization percentage
    # Approximation: each connection links ~2 regions
    regions_touched = min(dominant_count * 2, num_regions)
    sync_pct = (regions_touched / num_regions) * 100
    
    if sync_pct >= 60:
        recommendation = "THEMATIC"
    elif sync_pct >= 40:
        recommendation = "HYBRID"
    else:
        recommendation = "REGIONAL"
    
    return {
        "synchronization_pct": round(sync_pct),
        "dominant_theme": dominant_theme,
        "theme_connections": dominant_count,
        "recommendation": recommendation,
        "explanation": f"{dominant_theme} appears in {dominant_count} connections, touching ~{regions_touched} regions"
    }


def calculate_pic_score(
    probability: int,
    impact: int,
    confidence: int,
) -> tuple[PICScore, str]:
    """
    Calculate PIC score and decision.
    
    Returns:
        Tuple of (PICScore object, decision string)
    """
    pic = PICScore(
        probability=probability,
        impact=impact,
        confidence=confidence,
        total=(probability * impact * confidence) / 100,
        decision="PUBLISH" if (probability * impact * confidence) / 100 >= 60 else "KILL",
    )
    
    return pic, pic.decision


def apply_delta_test(
    analyst_output: AnalystOutput,
    previous_assessments: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """
    Apply the Delta Test: Does this change the 6-month forecast?
    
    Args:
        analyst_output: The analysis to evaluate
        previous_assessments: Optional dict of previous 3rd order predictions
        
    Returns:
        Tuple of (should_publish: bool, reasoning: str)
    """
    third_order = analyst_output.futures_wheel.third_order.lower()
    
    # Check for structural keywords
    structural_keywords = [
        "shift", "change", "alter", "realign", "restructure",
        "collapse", "emerge", "transform", "redefine", "break",
        "new", "unprecedented", "first time", "permanent",
    ]
    
    has_structural_change = any(kw in third_order for kw in structural_keywords)
    
    # Check confidence
    high_confidence = analyst_output.confidence in ["HIGH", "MODERATE"]
    
    if has_structural_change and high_confidence:
        return True, "Third-order effect indicates structural change"
    elif has_structural_change:
        return False, "Structural change claimed but low confidence"
    else:
        return False, "No significant change to 6-month forecast"
