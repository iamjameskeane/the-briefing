"""
Pydantic schemas for The Briefing Multi-Agent Pipeline.

These schemas enforce structured reasoning and prevent satisficing behavior
by requiring explicit analytical components at each stage.

v2: Analyst, Writer, Critic
Also includes Architect, Structure, Stylist (see state.py for new schemas)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


# =============================================================================
# ANALYST AGENT SCHEMAS
# =============================================================================


class ConstraintSet(BaseModel):
    """
    Immutable constraints that limit an actor's options.
    Based on Stratfor methodology: geography, economics, politics.
    """

    geographic: str = Field(
        ...,
        description="Physical/geographic factors limiting options (borders, terrain, resources)",
    )
    economic: str = Field(
        ...,
        description="Economic constraints (trade dependencies, debt, sanctions, resource control)",
    )
    political: str = Field(
        ...,
        description="Domestic political pressures (public opinion, elections, coalition dynamics)",
    )


class ActorAnalysis(BaseModel):
    """
    Constraints-of-Thought (Const-o-T) analysis for a single actor.
    Forces explicit reasoning about what actors CANNOT do.
    """

    actor: str = Field(
        ...,
        description="The actor (use metonyms: 'Washington', 'Beijing', not 'Biden', 'Xi')",
    )
    intent: str = Field(..., description="What this actor wants to achieve")
    constraints: ConstraintSet = Field(
        ..., description="Immutable constraints limiting this actor"
    )
    likely_action: str = Field(
        ...,
        description="Predicted action given constraints (use Sherman Kent probability)",
    )


class FuturesWheel(BaseModel):
    """
    Causal chain analysis forcing 1st, 2nd, and 3rd order effects.
    The 3rd order effect is the 'So What?' that makes analysis insightful.
    """

    driver_event: str = Field(
        ..., description="The primary event driving consequences"
    )
    driver_event_id: str = Field(..., description="Event ID for linking")
    first_order: str = Field(
        ..., description="Immediate consequence (what happens tomorrow)"
    )
    second_order: str = Field(
        ..., description="Regional reaction (what happens in 30 days)"
    )
    third_order: str = Field(
        ...,
        description="Structural shift (what changes in 6 months) - THE 'SO WHAT?'",
    )


class CompetingHypothesis(BaseModel):
    """
    Analysis of Competing Hypotheses (ACH) to mitigate confirmation bias.
    Forces acknowledgment of contradicting evidence.
    """

    consensus: str = Field(..., description="H1: The main assessment (consensus view)")
    contrarian: str = Field(
        ..., description="H2: The opposite view (what if we're wrong?)"
    )
    contradicting_evidence: list[str] = Field(
        ..., description="Facts that support H2 over H1"
    )
    evidence_event_ids: list[str] = Field(
        ..., description="Event IDs for the contradicting evidence"
    )


class AnalystOutput(BaseModel):
    """
    Complete output from the Analyst Agent.
    Pure structured reasoning - NO prose allowed.
    """

    region: str = Field(..., description="Region being analyzed")

    # Step-Back Abstraction
    geopolitical_archetype: str = Field(
        ...,
        description="Dominant framework (e.g., 'Security Dilemma', 'Resource Curse', 'Thucydides Trap')",
    )
    archetype_explanation: str = Field(
        ..., description="How current events manifest this pattern"
    )

    # Const-o-T
    primary_actors: list[ActorAnalysis] = Field(
        ..., description="2-4 major actors with constraints analysis"
    )

    # Futures Wheel
    futures_wheel: FuturesWheel = Field(..., description="Causal chain to 3rd order")

    # ACH
    competing_hypotheses: CompetingHypothesis = Field(
        ..., description="Consensus vs contrarian views"
    )

    # PMESII-PT tagging
    pmesii_tags: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Events tagged by dimension: Political, Military, Economic, Social, Infrastructure, Information",
    )

    # Confidence
    confidence: Literal["HIGH", "MODERATE", "LOW"] = Field(
        ..., description="Confidence in assessment"
    )
    confidence_rationale: str = Field(
        ..., description="Why this confidence level (evidence quality, source diversity)"
    )
    
    # External Sources (from Tavily searches)
    external_sources: list[dict] = Field(
        default_factory=list,
        description="External sources used from web search: [{'title': '...', 'url': '...', 'publisher': '...'}]"
    )


@dataclass
class AnalystInput:
    """Input data for the Analyst Agent.
    
    Can work in two modes:
    1. REGIONAL (legacy): Analyze events for a specific region
    2. THEMATIC: Analyze a thematic cluster with function calls for context
    """
    
    # Identification
    cluster_id: str = ""  # Thematic cluster ID or region name
    theme_label: str = ""  # Human-readable theme label (e.g., "US Policy Shifts")
    
    # Events to analyze
    events: list[dict] = None  # Raw event data
    
    # Context (optional - Analyst can call tools to get context)
    regions_touched: list[str] = None  # Which regions this cluster touches
    countries_mentioned: list[str] = None  # Countries extracted from events
    
    # Legacy fields (for backward compatibility)
    region: str = ""  # Deprecated: use cluster_id
    region_constraints: dict = None  # Deprecated: use function calls
    is_featured: bool = False
    
    def __post_init__(self):
        if self.events is None:
            self.events = []
        if self.regions_touched is None:
            self.regions_touched = []
        if self.countries_mentioned is None:
            self.countries_mentioned = []
        if self.region_constraints is None:
            self.region_constraints = {}
        # If only region is set (legacy mode), use it as cluster_id
        if self.region and not self.cluster_id:
            self.cluster_id = self.region
            self.theme_label = self.region


# =============================================================================
# WRITER AGENT SCHEMAS
# =============================================================================


class WriterOutput(BaseModel):
    """Output from the Writer Agent."""

    content: str = Field(..., description="The formatted markdown content")
    word_count: int = Field(..., description="Actual word count")
    entity_count: int = Field(
        ..., description="Number of specific entities (names, numbers, dates)"
    )


@dataclass
class WriterInput:
    """Input data for the Writer Agent."""

    analyst_output: AnalystOutput
    section_type: Literal["featured", "regional", "sleeper"]
    word_limit: int  # 500 for featured, 250 for regional, 100 for sleeper
    events_for_linking: list[dict]  # For deep links
    revision_feedback: str | None = None  # From ContentCritic if retry
    previous_draft: str | None = None  # Previous draft for retry context
    section_blueprint: "SectionBlueprint | None" = None  # structured plan from Structure agent
    apply_chain_of_density: bool = True  # iterative density improvement (skip for sleeper)
    sacred_elements: "SacredElements | None" = None  # facts Writer must preserve exactly
    editorial_angle: str | None = None  # Architect's editorial framing for featured stories


# =============================================================================
# PIPELINE SCHEMAS
# =============================================================================


@dataclass
class SectionOutput:
    """Output from the full three-agent pipeline for one section."""

    region: str
    content: str  # Final markdown
    analyst_data: AnalystOutput  # For debugging/audit
    critic_score: int
    quality_warning: bool = False  # True if max retries exceeded


@dataclass
class RegionContext:
    """Pre-computed context for a region (from constraints.py)."""

    geography: str
    economics: str
    politics: str
    key_relationships: dict[str, str] = field(default_factory=dict)
    historical_patterns: list[str] = field(default_factory=list)
    critical_infrastructure: list[str] = field(default_factory=list)

    def to_prompt_string(self) -> str:
        """Format for injection into prompts."""
        lines = [
            "## REGION CONTEXT (Pre-computed constraints)",
            "",
            "### Geographic Constraints",
            self.geography,
            "",
            "### Economic Constraints",
            self.economics,
            "",
            "### Political Constraints",
            self.politics,
            "",
        ]

        if self.key_relationships:
            lines.append("### Key Relationships")
            for actor, relationship in self.key_relationships.items():
                lines.append(f"- **{actor}**: {relationship}")
            lines.append("")

        if self.historical_patterns:
            lines.append("### Historical Patterns")
            for pattern in self.historical_patterns:
                lines.append(f"- {pattern}")
            lines.append("")

        if self.critical_infrastructure:
            lines.append("### Critical Infrastructure (Reference in 3rd Order Effects)")
            for infra in self.critical_infrastructure:
                lines.append(f"- {infra}")
            lines.append("")

        return "\n".join(lines)
