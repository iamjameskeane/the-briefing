"""
Document Assembly for The Briefing.

Combines all generated sections into the final briefing document.
Handles:
- Executive summary generation
- Section ordering
- Cross-regional connections
- Calendar integration
- Metadata generation
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from agents.schemas import SectionOutput
from config import get_config
from utils import logger

if TYPE_CHECKING:
    pass


# =============================================================================
# DOCUMENT TEMPLATES
# =============================================================================

DOCUMENT_HEADER = """# STATE OF THE WORLD
## Weekly Intelligence Briefing

**Week of {week_date}** | **{event_count} events analyzed** | **{region_count} regions covered**

---

"""

EXECUTIVE_SUMMARY_TEMPLATE = """## EXECUTIVE SUMMARY

{summary_content}

---

"""

CROSS_REGIONAL_TEMPLATE = """## CROSS-REGIONAL CONNECTIONS

{connections_content}

---

"""

CALENDAR_TEMPLATE = """## LOOKING AHEAD

Key events to watch in the coming weeks:

{calendar_content}

---

"""

METHODOLOGY_TEMPLATE = """## METHODOLOGY

This briefing was generated using The Briefing multi-agent analysis pipeline:

- **{event_count}** events processed from {source_count} sources
- **{region_count}** regions analyzed with dedicated Analyst, Writer, and Critic agents
- Analysis frameworks: Constraints-of-Thought, Futures Wheel, Analysis of Competing Hypotheses
- Probability language follows Sherman Kent doctrine

*Generated: {timestamp}*
"""


# =============================================================================
# ASSEMBLY FUNCTIONS
# =============================================================================


def generate_executive_summary(
    featured_section: SectionOutput,
    regional_sections: list[SectionOutput],
) -> str:
    """
    Generate the executive summary from section outputs.

    The executive summary highlights:
    1. The featured topic's key judgment
    2. Top 3-4 regional developments
    3. Key cross-cutting themes
    """
    config = get_config()

    # Extract key judgments
    summary_parts = []

    # Featured topic lead
    featured_archetype = featured_section.analyst_data.geopolitical_archetype
    featured_third_order = featured_section.analyst_data.futures_wheel.third_order
    summary_parts.append(
        f"**This week's featured analysis** examines a classic {featured_archetype} "
        f"dynamic in {featured_section.region}. The key implication: {featured_third_order}"
    )

    # Top regional developments (by critic score)
    sorted_regionals = sorted(
        regional_sections,
        key=lambda s: s.critic_score,
        reverse=True,
    )

    summary_parts.append("\n\n**Regional highlights:**\n")
    for section in sorted_regionals[:4]:
        third_order = section.analyst_data.futures_wheel.third_order
        # Truncate at word boundary to avoid mid-word cutoff
        if len(third_order) > 150:
            truncated = third_order[:150].rsplit(" ", 1)[0] + "..."
        else:
            truncated = third_order
        summary_parts.append(f"- **{section.region}**: {truncated}")

    return "".join(summary_parts)


def format_cross_regional_connections(
    connections: list[dict],
) -> str:
    """Format cross-regional connections for the briefing.
    
    Handles CrossRegionalConnection.to_dict() format:
    - regions: list[str] - the connected regions
    - connection_type: str - type of connection
    - explanation: str - description of the link
    """
    if not connections:
        return "*No significant cross-regional connections identified this week.*"

    lines = []
    for conn in connections[:5]:  # Max 5 connections
        # Handle CrossRegionalConnection format
        regions = conn.get("regions", [])
        if len(regions) >= 2:
            region1 = regions[0].replace("_", " ").title()
            region2 = regions[1].replace("_", " ").title()
        else:
            region1 = "Unknown"
            region2 = "Unknown"
        
        connection_type = conn.get("connection_type", "").replace("_", " ").title()
        explanation = conn.get("explanation", "Related developments")
        
        lines.append(f"- **{region1} ↔ {region2}** ({connection_type}): {explanation}")

    return "\n".join(lines)


def format_calendar_events(
    calendar_events: list,
) -> str:
    """Format upcoming calendar events.
    
    Handles both dict and UpcomingEvent objects.
    """
    if not calendar_events:
        return "*No major scheduled events in the coming weeks.*"

    lines = []
    for event in calendar_events[:6]:
        # Handle both dict and UpcomingEvent objects
        if hasattr(event, "date"):
            # UpcomingEvent object
            date = event.date
            title = event.event if hasattr(event, "event") else "Unknown event"
            significance = event.likely_outcome if hasattr(event, "likely_outcome") else ""
        else:
            # dict
            date = event.get("date", "TBD")
            title = event.get("title", event.get("event", "Unknown event"))
            significance = event.get("significance", event.get("likely_outcome", ""))
        
        line = f"- **{date}**: {title}"
        if significance:
            line += f" — *{significance}*"
        lines.append(line)

    return "\n".join(lines)


def assemble_document(
    featured_section: SectionOutput,
    regional_sections: list[SectionOutput],
    cross_regional_connections: list[dict] | None = None,
    calendar_events: list[dict] | None = None,
    event_count: int = 0,
    source_count: int = 0,
) -> str:
    """
    Assemble the complete briefing document.

    Args:
        featured_section: The featured analysis section
        regional_sections: List of regional briefing sections
        cross_regional_connections: Optional cross-regional links
        calendar_events: Optional upcoming events
        event_count: Total events processed
        source_count: Number of sources

    Returns:
        Complete markdown document
    """
    logger.info("📝 Assembling final document...")

    now = datetime.now(timezone.utc)
    week_date = now.strftime("%B %d, %Y")
    timestamp = now.isoformat()

    region_count = len(regional_sections) + 1  # +1 for featured

    # Build document
    doc_parts = []

    # Header
    doc_parts.append(
        DOCUMENT_HEADER.format(
            week_date=week_date,
            event_count=event_count,
            region_count=region_count,
        )
    )

    # Executive Summary
    exec_summary = generate_executive_summary(featured_section, regional_sections)
    doc_parts.append(
        EXECUTIVE_SUMMARY_TEMPLATE.format(summary_content=exec_summary)
    )

    # Featured Analysis
    doc_parts.append(featured_section.content)
    doc_parts.append("\n\n---\n\n")

    # Regional Briefings
    doc_parts.append("## REGIONAL BRIEFINGS\n\n")

    # Sort by a logical order (can customize)
    region_order = [
        "MIDDLE_EAST",
        "EUROPE",
        "EAST_ASIA",
        "SOUTH_ASIA",
        "SOUTHEAST_ASIA",
        "RUSSIA_EURASIA",
        "AFRICA_SUB_SAHARAN",
        "LATIN_AMERICA",
        "CENTRAL_ASIA",
        "OCEANIA",
    ]

    # Sort regionals by predefined order, then alphabetically
    def sort_key(s: SectionOutput) -> tuple:
        try:
            idx = region_order.index(s.region.upper().replace(" ", "_"))
        except ValueError:
            idx = 999
        return (idx, s.region)

    sorted_sections = sorted(regional_sections, key=sort_key)

    for section in sorted_sections:
        doc_parts.append(section.content)
        doc_parts.append("\n\n")

    doc_parts.append("---\n\n")

    # Cross-Regional Connections
    if cross_regional_connections:
        connections_content = format_cross_regional_connections(cross_regional_connections)
        doc_parts.append(
            CROSS_REGIONAL_TEMPLATE.format(connections_content=connections_content)
        )

    # Calendar
    if calendar_events:
        calendar_content = format_calendar_events(calendar_events)
        doc_parts.append(
            CALENDAR_TEMPLATE.format(calendar_content=calendar_content)
        )

    # Methodology footer
    doc_parts.append(
        METHODOLOGY_TEMPLATE.format(
            event_count=event_count,
            source_count=source_count,
            region_count=region_count,
            timestamp=timestamp,
        )
    )

    document = "".join(doc_parts)
    logger.info(f"   ✅ Document assembled: {len(document)} characters, {len(document.split())} words")

    return document


def generate_metadata(
    featured_section: SectionOutput,
    regional_sections: list[SectionOutput],
    event_count: int = 0,
) -> dict:
    """
    Generate metadata for the briefing (for storage/indexing).
    """
    now = datetime.now(timezone.utc)

    return {
        "id": now.strftime("%Y-%m-%d"),
        "generated_at": now.isoformat(),
        "week_of": now.strftime("%B %d, %Y"),
        "event_count": event_count,
        "region_count": len(regional_sections) + 1,
        "featured_region": featured_section.region,
        "featured_archetype": featured_section.analyst_data.geopolitical_archetype,
        "regions": [s.region for s in regional_sections],
        "quality_scores": {
            featured_section.region: featured_section.critic_score,
            **{s.region: s.critic_score for s in regional_sections},
        },
        "average_quality": sum(
            [featured_section.critic_score] + [s.critic_score for s in regional_sections]
        ) / (len(regional_sections) + 1),
    }
