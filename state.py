"""
The Briefing - Pipeline State Management.

This module defines the shared state that flows through all agents in the pipeline.
Each phase mutates specific portions of the state, enabling:
- Clean handoffs between agents
- Checkpointing and resumption
- Debugging and auditing

Architecture:
    PipelineState is the single source of truth passed through:
    Analyst → Architect → Structure → Writer → Stylist → Critic → Assembler
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# SUPPORTING TYPES
# =============================================================================

@dataclass
class SourceDocument:
    """A source document used in analysis (event, article, etc.)."""
    
    id: str
    title: str
    source: str
    content: str
    published_at: datetime
    region: str
    url: Optional[str] = None


@dataclass
class RegionCluster:
    """A cluster of events for a specific region."""
    
    region: str
    events: list[dict]
    storylines: list[str]
    importance_score: float = 0.0


# =============================================================================
# ARCHITECT OUTPUT SCHEMAS
# =============================================================================

class FeaturedPlan(BaseModel):
    """Plan for the featured analysis section."""
    
    region: str = Field(..., description="Region for featured analysis")
    source_cluster_id: Optional[str] = Field(
        default=None,
        description="Original cluster ID or theme label - used to match back to analyst data"
    )
    headline: str = Field(..., description="Punchy title, not just region name")
    angle: str = Field(..., description="The 'nub' - unique angle for this story")
    story_archetype: str = Field(
        default="CRISIS",
        description="Story archetype: CRISIS, TREND, PIVOT, SLEEPER, COMPETITION, CONSTRAINT, LEADER"
    )
    word_target: int = Field(default=600, description="Target word count")
    rationale: str = Field(..., description="Why this deserves featured treatment")


class SectionDecision(BaseModel):
    """Architect's decision for a single section."""
    
    region: str = Field(..., description="Region name")
    source_cluster_id: Optional[str] = Field(
        default=None,
        description="Original cluster ID or theme label - used to match back to analyst data"
    )
    headline: str = Field(..., description="Section title (not just region name)")
    treatment: str = Field(
        ..., 
        description="FULL, SHORT, COMBINED, SLEEPER, QUICK_HIT, KILL"
    )
    archetype: str = Field(
        ...,
        description="Story archetype: CRISIS, TREND, PIVOT, SLEEPER, COMPETITION, CONSTRAINT, LEADER"
    )
    word_target: int = Field(..., description="Target word count for this section")
    combine_with: Optional[str] = Field(
        default=None,
        description="If COMBINED treatment, which region to combine with"
    )
    position_rationale: str = Field(..., description="Why this section is in this position")
    rationale: str = Field(..., description="Why this treatment was chosen")


class KilledRegion(BaseModel):
    """A region that was killed (not included in briefing)."""
    
    region: str = Field(..., description="Region name")
    reason: str = Field(
        ..., 
        description="NOTHING_NEW, BELOW_THRESHOLD, REDUNDANT, or HOLD_FOR_NEXT_WEEK"
    )
    pic_score: float = Field(..., description="PIC score that led to kill decision")
    events_count: int = Field(default=0, description="Number of events in this region")
    highest_severity: int = Field(default=0, description="Highest event severity in this region")
    brief_explanation: str = Field(default="", description="Brief explanation of kill decision")


class QuickHit(BaseModel):
    """A quick hit item (1-2 sentences, no deep analysis)."""
    
    region: str = Field(..., description="Region name")
    headline: str = Field(..., description="One-line headline (max 25 words)")
    content: str = Field(..., description="1-2 sentence summary")
    event_id: str = Field(..., description="Primary event ID for linking")
    event_ids: list[str] = Field(default_factory=list, description="All source event IDs")


class SectionGroup(BaseModel):
    """A group of sections under a shared header."""
    
    title: str = Field(..., description="Evocative group header (e.g., 'The Pressure Builds', 'On the Radar')")
    sections: list[str] = Field(..., description="Ordered list of region names in this group")
    group_rationale: str = Field(default="", description="Why these sections are grouped together")


class DocumentSkeleton(BaseModel):
    """
    The Architect's complete plan for the document.
    This is the editorial blueprint that all downstream agents follow.
    
    CRITICAL DISTINCTION:
    - featured = The single most important REGION story (Spoke 1 in THEMATIC mode)
    - hub_mechanism = The ABSTRACT MECHANISM connecting multiple regions (only for THEMATIC mode)
    
    In THEMATIC mode:
    - The hub is NOT a region or story—it's the underlying FORCE/PATTERN
    - The hub must be ABSTRACT (could apply to any actor/context)
    - The featured region is the first SPOKE (manifestation of the hub mechanism)
    - Example: Hub = "Economic Coercion as Statecraft", Featured = "Trump's Greenland Gambit"
    """
    
    # Main content
    featured: FeaturedPlan = Field(..., description="Featured analysis plan (single most important region)")
    sections: list[SectionDecision] = Field(
        ..., 
        description="Ordered list of section decisions (excluding killed)"
    )
    killed: list[KilledRegion] = Field(
        default_factory=list,
        description="Regions killed by the Architect"
    )
    quick_hits: list[QuickHit] = Field(
        default_factory=list,
        description="Quick hit items"
    )
    
    # Structure
    narrative_arc: str = Field(
        ...,
        description="One-sentence description of the week's through-line"
    )
    narrative_arc_explanation: str = Field(
        ...,
        description="Why this narrative arc was chosen"
    )
    section_groups: list[SectionGroup] = Field(
        ...,
        description="How sections are grouped under headers"
    )
    transitions: dict[str, str] = Field(
        default_factory=dict,
        description="Transition text between sections. Key: 'RegionA → RegionB', Value: transition or 'HARD_BREAK'"
    )
    organizing_principle: str = Field(
        ...,
        description="REGIONAL, THEMATIC, or HYBRID"
    )
    
    # Hub-and-Spoke fields (THEMATIC mode only)
    hub_mechanism: Optional[str] = Field(
        default=None,
        description="For THEMATIC mode: The ABSTRACT mechanism connecting regions (NOT an actor or story). Must be universal, not actor-specific. Examples: 'Economic Coercion as Statecraft' (NOT 'Trump's Trade Policy'), 'Unilateral Action Replacing Consensus' (NOT 'US Foreign Policy Shift')"
    )
    hub_explanation: Optional[str] = Field(
        default=None,
        description="For THEMATIC mode: 2-3 sentences explaining the MECHANISM itself (not summarizing a story). Should explain the force/pattern operating across multiple spokes."
    )
    hub_manifestations: Optional[list[str]] = Field(
        default=None,
        description="For THEMATIC mode: List of region names that demonstrate this mechanism (the spokes)"
    )
    hub_angle: Optional[str] = Field(
        default=None,
        description="For THEMATIC mode: The angle type for the hub mechanism (MISSING_THE_POINT, UNINTENDED_CONSEQUENCE, etc.)"
    )


# =============================================================================
# STRUCTURE AGENT SCHEMAS
# =============================================================================

class EventEmphasis(BaseModel):
    """How much to emphasize a specific event in the section."""
    
    event_id: str = Field(..., description="Event ID from source data")
    event_title: str = Field(default="", description="Event title for reference")
    emphasis: str = Field(..., description="LEAD, SUPPORT, or MENTION")
    how_to_use: str = Field(default="", description="How to use this event (e.g., 'Use as pivot point')")
    role: str = Field(default="", description="What role this event plays in the narrative")


class ParagraphPlan(BaseModel):
    """Plan for a single paragraph in a section."""
    
    purpose: str = Field(..., description="What this paragraph accomplishes")
    word_target: int = Field(..., description="Target word count")
    key_facts: list[str] = Field(
        default_factory=list,
        description="Specific facts that must appear"
    )
    beat: str = Field(
        ...,
        description="Beat from the archetype template (e.g., 'What just happened', 'Why it matters NOW')"
    )


class SectionBlueprint(BaseModel):
    """
    Structure Agent's detailed plan for a single section.
    The Writer follows this paragraph-by-paragraph.
    """
    
    region: str = Field(..., description="Region name")
    archetype: str = Field(..., description="Story archetype")
    word_target: int = Field(..., description="Total section word target")
    
    # Opening
    hook_type: str = Field(
        ...,
        description="STATEMENT, QUESTION, SURPRISE, CONTRAST, or SCENE"
    )
    hook_draft: str = Field(
        ...,
        description="Draft of the opening hook (Writer can refine)"
    )
    
    # Body
    paragraphs: list[ParagraphPlan] = Field(
        ...,
        description="Paragraph-by-paragraph plan"
    )
    key_events: list[EventEmphasis] = Field(
        default_factory=list,
        description="Which events to emphasize"
    )
    
    # Closing
    closing_type: str = Field(
        ...,
        description="PREDICTION, WARNING, QUESTION, CALLBACK, or IMAGE"
    )
    closing_draft: str = Field(
        ...,
        description="Draft of the closing (Writer can refine)"
    )


# =============================================================================
# CRITIC SCHEMAS (ENHANCED WITH CoVe)
# =============================================================================

class FactualClaim(BaseModel):
    """A factual claim extracted from the draft for verification."""
    
    id: str = Field(..., description="Unique claim ID")
    claim_text: str = Field(..., description="The claim as stated in the draft")
    claim_type: str = Field(
        ...,
        description="STATISTIC, DATE, NAME, EVENT, QUOTE, RELATIONSHIP"
    )
    source_sentence: str = Field(..., description="Full sentence containing the claim")


class VerificationResult(BaseModel):
    """Result of verifying a single claim against sources."""
    
    claim_id: str = Field(..., description="ID of the claim being verified")
    verification_question: str = Field(
        ...,
        description="Question asked to verify the claim"
    )
    independent_answer: str = Field(
        ...,
        description="Answer derived from sources (without seeing draft)"
    )
    draft_answer: str = Field(
        ...,
        description="What the draft claims"
    )
    match: bool = Field(..., description="Whether draft matches sources")
    confidence: str = Field(
        default="HIGH",
        description="Confidence in verification: HIGH, MEDIUM, LOW"
    )


class HallucinationFlag(BaseModel):
    """A flagged hallucination in the draft."""
    
    claim_id: str = Field(..., description="ID of the claim flagged")
    severity: str = Field(
        ...,
        description="CRITICAL (wrong facts), MINOR (embellishment), WARNING (unverifiable)"
    )
    explanation: str = Field(..., description="What's wrong and how to fix it")


class CoVeResult(BaseModel):
    """Complete Chain of Verification result."""
    
    claims_extracted: list[FactualClaim] = Field(
        default_factory=list,
        description="All factual claims found in draft"
    )
    verification_results: list[VerificationResult] = Field(
        default_factory=list,
        description="Verification results for each claim"
    )
    hallucination_flags: list[HallucinationFlag] = Field(
        default_factory=list,
        description="Flagged hallucinations"
    )
    
    @property
    def has_critical_hallucinations(self) -> bool:
        """True if any critical hallucinations were found."""
        return any(f.severity == "CRITICAL" for f in self.hallucination_flags)
    
    @property
    def hallucination_rate(self) -> float:
        """Percentage of claims that failed verification."""
        if not self.verification_results:
            return 0.0
        failed = sum(1 for r in self.verification_results if not r.match)
        return failed / len(self.verification_results)


class ContentCriticResult(BaseModel):
    """
    Result from ContentCritic - evaluates factual/analytical quality.
    Used in the Writer loop.
    """
    
    passed: bool = Field(..., description="Did content pass threshold?")
    score: int = Field(..., ge=0, le=100, description="Content score (0-100)")
    
    # Individual check scores
    angle_delivery_score: int = Field(default=0, description="Did opening execute angle?")
    prediction_score: int = Field(default=0, description="Forward-looking assessments?")
    constraints_score: int = Field(default=0, description="Actor constraints stated?")
    density_score: int = Field(default=0, description="Entity density in prose?")
    sherman_kent_score: int = Field(default=0, description="Uses probability terms?")
    coverage_score: int = Field(default=0, description="Key insights from analyst?")
    sacred_elements_score: int = Field(default=0, description="Facts preserved exactly?")
    
    # CoVe
    cove_result: CoVeResult = Field(
        default_factory=CoVeResult,
        description="Chain of Verification results"
    )
    
    # Issues and feedback
    issues: list[str] = Field(
        default_factory=list,
        description="Specific content issues found"
    )
    feedback: str = Field(
        default="",
        description="Formatted feedback for Writer revision"
    )


class StyleCriticResult(BaseModel):
    """
    Result from StyleCritic - evaluates prose quality.
    Used in the Stylist loop.
    """
    
    passed: bool = Field(..., description="Did style pass threshold?")
    score: int = Field(..., ge=0, le=100, description="Style score (0-100)")
    
    # Individual check scores
    weasel_free_score: int = Field(default=0, description="No banned phrases?")
    sentence_variety_score: int = Field(default=0, description="Varied sentence openings?")
    active_voice_score: int = Field(default=0, description="Minimal passive voice?")
    opening_punch_score: int = Field(default=0, description="Strong first sentence?")
    closing_resonance_score: int = Field(default=0, description="Memorable ending?")
    quotable_line_score: int = Field(default=0, description="One punchy phrase?")
    orwell_score: int = Field(default=0, description="Short words, no jargon?")
    fact_preservation_score: int = Field(default=0, description="Writer facts preserved?")
    
    # Content regression penalty (negative when Stylist loses analytical content)
    content_regression_penalty: int = Field(
        default=0,
        le=0,
        description="Penalty for lost predictions/constraints/Sherman Kent terms (≤0)"
    )
    
    # Issues and feedback
    issues: list[str] = Field(
        default_factory=list,
        description="Specific style issues found"
    )
    feedback: str = Field(
        default="",
        description="Formatted feedback for Stylist revision"
    )


class CriticResult(BaseModel):
    """
    Combined Critic result (legacy compatibility).
    Contains both content and style results.
    """
    
    passed: bool = Field(..., description="Overall pass/fail")
    overall_score: int = Field(..., ge=0, le=100, description="Total score (0-100)")
    
    # Content evaluation
    content_passed: bool = Field(..., description="Did content pass?")
    content_score: int = Field(..., ge=0, le=50, description="Content score (0-50)")
    content_issues: list[str] = Field(
        default_factory=list,
        description="Specific content issues found"
    )
    
    # Style evaluation
    style_passed: bool = Field(..., description="Did style pass?")
    style_score: int = Field(..., ge=0, le=50, description="Style score (0-50)")
    style_issues: list[str] = Field(
        default_factory=list,
        description="Specific style issues found"
    )
    
    # CoVe
    cove_result: CoVeResult = Field(
        default_factory=CoVeResult,
        description="Chain of Verification results"
    )
    
    # Feedback routing
    feedback_for_writer: Optional[str] = Field(
        default=None,
        description="Specific feedback for Writer (content issues)"
    )
    feedback_for_stylist: Optional[str] = Field(
        default=None,
        description="Specific feedback for Stylist (style issues)"
    )
    
    # New: references to individual results
    content_result: Optional[ContentCriticResult] = Field(
        default=None,
        description="Full content critic result"
    )
    style_result: Optional[StyleCriticResult] = Field(
        default=None,
        description="Full style critic result"
    )


# =============================================================================
# SACRED ELEMENTS (Entity Preservation)
# =============================================================================

class SacredElements(BaseModel):
    """
    Facts that the Writer/Stylist cannot fabricate or alter.
    Extracted from Analyst output and enforced downstream.
    """
    
    proper_nouns: list[str] = Field(
        default_factory=list,
        description="Named entities that must be preserved exactly"
    )
    statistics: list[str] = Field(
        default_factory=list,
        description="Numbers and percentages from sources"
    )
    dates: list[str] = Field(
        default_factory=list,
        description="Specific dates from sources"
    )
    quotes: list[str] = Field(
        default_factory=list,
        description="Direct quotes that cannot be paraphrased"
    )
    event_ids: list[str] = Field(
        default_factory=list,
        description="Event IDs from sources"
    )


# =============================================================================
# PIC MATRIX (Probability × Impact × Confidence)
# =============================================================================

class PICScore(BaseModel):
    """
    Probability × Impact × Confidence scoring for editorial decisions.
    Used by Architect to decide KILL/HOLD/PUBLISH.
    """
    
    probability: int = Field(..., ge=1, le=5, description="How likely is this significant? (1-5)")
    impact: int = Field(..., ge=1, le=5, description="How much does this matter? (1-5)")
    confidence: int = Field(..., ge=1, le=5, description="How confident are we? (1-5)")
    
    @property
    def score(self) -> float:
        """Calculate PIC score: (P × I × C) / 5."""
        return (self.probability * self.impact * self.confidence) / 5
    
    @property
    def decision(self) -> str:
        """
        Editorial decision based on score.
        - < 5: KILL
        - 5-15: HOLD (Quick Hit or Sleeper)
        - > 15: PUBLISH (Full or Short)
        """
        if self.score < 5:
            return "KILL"
        elif self.score <= 15:
            return "HOLD"
        return "PUBLISH"


# =============================================================================
# PIPELINE STATE
# =============================================================================

@dataclass
class PipelineState:
    """
    Shared state for The Briefing pipeline.
    Mutated by each phase, enabling clean handoffs and checkpointing.
    """
    
    # --- Immutable Inputs ---
    run_id: str
    started_at: datetime
    source_context: list[SourceDocument] = field(default_factory=list)
    event_clusters: dict[str, RegionCluster] = field(default_factory=dict)
    
    # --- Phase 1 Metadata ---
    total_event_count: int = 0  # Total events analyzed
    total_source_count: int = 0  # Number of unique sources
    
    # --- Phase 1: Analysis ---
    # Import AnalystOutput from schemas.py to avoid duplication
    analyst_outputs: dict[str, "AnalystOutput"] = field(default_factory=dict)
    
    # --- Phase 2: Editorial (Editor + Architect) ---
    editor_decisions: Optional["EditorDecisions"] = None  # Editor's research and decisions
    architect_reasoning: Optional[str] = None  # Architect's step 1 reasoning output (before JSON formatting)
    document_skeleton: Optional[DocumentSkeleton] = None  # Architect's structured output
    pic_scores: dict[str, PICScore] = field(default_factory=dict)
    
    # --- Phase 3: Structure ---
    section_blueprints: dict[str, SectionBlueprint] = field(default_factory=dict)
    
    # --- Phase 4: Drafting ---
    writer_drafts: dict[str, str] = field(default_factory=dict)
    sacred_elements: dict[str, SacredElements] = field(default_factory=dict)
    
    # --- Phase 4b: Styling ---
    styled_drafts: dict[str, str] = field(default_factory=dict)
    
    # --- Phase 5: Critique ---
    critique_results: dict[str, CriticResult] = field(default_factory=dict)
    retry_counts: dict[str, int] = field(default_factory=dict)
    
    # --- Phase 6: Assembly ---
    final_output: Optional[str] = None
    
    # --- Metadata ---
    checkpoints: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def add_checkpoint(self, checkpoint_name: str) -> None:
        """Record a checkpoint."""
        self.checkpoints.append(f"{datetime.now().isoformat()}: {checkpoint_name}")
    
    def add_error(self, error: str) -> None:
        """Record an error."""
        self.errors.append(f"{datetime.now().isoformat()}: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Record a warning."""
        self.warnings.append(f"{datetime.now().isoformat()}: {warning}")
    
    @property
    def publishable_regions(self) -> list[str]:
        """Regions that passed the Architect's kill decision."""
        if not self.document_skeleton:
            return list(self.analyst_outputs.keys())
        return [s.region for s in self.document_skeleton.sections]
    
    @property
    def killed_regions(self) -> list[str]:
        """Regions killed by the Architect."""
        if not self.document_skeleton:
            return []
        return [k.region for k in self.document_skeleton.killed]
    
    def get_compacted_context_for_region(self, region: str, max_tokens: int = 4000) -> dict:
        """
        Get compacted context for a specific region.
        
        Implements context compaction from research:
        - Keeps only relevant source documents
        - Summarizes prior phase outputs
        - Preserves critical facts
        
        Args:
            region: The region to get context for
            max_tokens: Approximate token budget
            
        Returns:
            Compacted context dict
        """
        context = {
            "region": region,
            "run_id": self.run_id,
        }
        
        # Add analyst output (compact form)
        if region in self.analyst_outputs:
            ao = self.analyst_outputs[region]
            context["analyst_summary"] = {
                "archetype": ao.geopolitical_archetype,
                "third_order_effect": ao.futures_wheel.third_order,
                "key_constraints": [
                    f"{a.actor}: {a.constraints.geographic[:50]}"
                    for a in ao.primary_actors[:2]
                ],
                "confidence": ao.confidence,
            }
        
        # Add skeleton decision (compact form)
        if self.document_skeleton:
            for section in self.document_skeleton.sections:
                if section.region == region:
                    context["skeleton_decision"] = {
                        "treatment": section.treatment,
                        "archetype": section.archetype,
                        "word_target": section.word_target,
                    }
                    break
        
        # Add blueprint summary (compact form)
        if region in self.section_blueprints:
            bp = self.section_blueprints[region]
            context["blueprint_summary"] = {
                "archetype": bp.archetype,
                "hook_type": bp.hook_type,
                "num_paragraphs": len(bp.paragraphs),
                "closing_type": bp.closing_type,
            }
        
        # Add source events (limited)
        if region in self.event_clusters:
            cluster = self.event_clusters[region]
            context["top_events"] = [
                {"title": e.get("title", "")[:50], "severity": e.get("severity", 5)}
                for e in cluster.events[:5]
            ]
        
        # Add sacred elements
        if region in self.sacred_elements:
            se = self.sacred_elements[region]
            context["sacred_elements"] = {
                "nouns": se.proper_nouns[:5],
                "stats": se.statistics[:3],
                "dates": se.dates[:3],
            }
        
        return context
    
    def get_state_summary(self) -> str:
        """
        Get a human-readable summary of the pipeline state.
        Useful for debugging and auditing.
        """
        lines = [
            f"=== Pipeline State: {self.run_id} ===",
            f"Started: {self.started_at.isoformat()}",
            f"Checkpoints: {len(self.checkpoints)}",
            "",
            "Phase Progress:",
            f"  Analyst outputs: {len(self.analyst_outputs)} regions",
            f"  Skeleton: {'✓' if self.document_skeleton else '✗'}",
            f"  Blueprints: {len(self.section_blueprints)} sections",
            f"  Writer drafts: {len(self.writer_drafts)} sections",
            f"  Styled drafts: {len(self.styled_drafts)} sections",
            f"  Critique results: {len(self.critique_results)} sections",
            f"  Final output: {'✓' if self.final_output else '✗'}",
            "",
            f"Publishable: {self.publishable_regions}",
            f"Killed: {self.killed_regions}",
            "",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]
        
        if self.errors:
            lines.append("")
            lines.append("Latest Errors:")
            for e in self.errors[-3:]:
                lines.append(f"  - {e}")
        
        return "\n".join(lines)


# =============================================================================
# TYPE IMPORTS (for forward references)
# =============================================================================

# Import at module level to resolve forward references
# This is imported at the end to avoid circular imports
from agents.schemas import AnalystOutput  # noqa: E402, F401
