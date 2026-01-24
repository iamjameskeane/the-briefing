"""
Tests for The Briefing state and schema definitions.

Validates:
- PipelineState serialization/deserialization
- New schema validations (PICScore, DocumentSkeleton, SectionBlueprint, etc.)
- Example bank loading
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from state import (
    PipelineState,
    PICScore,
    DocumentSkeleton,
    SectionBlueprint,
    SectionDecision,
    FeaturedPlan,
    KilledRegion,
    QuickHit,
    SectionGroup,
    ParagraphPlan,
    EventEmphasis,
    CoVeResult,
    ContentCriticResult,
    StyleCriticResult,
    FactualClaim,
    VerificationResult,
    HallucinationFlag,
    SacredElements,
    SourceDocument,
    RegionCluster,
)
from examples import get_style_examples, ECONOMIST_EXAMPLES
from examples.negative_constraints import check_for_cliches, apply_brevity_substitutions


class TestPICScore:
    """Tests for PIC Matrix scoring."""
    
    def test_score_calculation(self):
        """Score should be (P × I × C) / 5."""
        pic = PICScore(probability=5, impact=5, confidence=5)
        assert pic.score == 25.0  # 125 / 5
        
        pic = PICScore(probability=1, impact=1, confidence=1)
        assert pic.score == 0.2  # 1 / 5
        
        pic = PICScore(probability=3, impact=4, confidence=2)
        assert pic.score == 4.8  # 24 / 5
    
    def test_kill_decision(self):
        """Score < 5 should return KILL."""
        pic = PICScore(probability=2, impact=2, confidence=2)
        assert pic.score == 1.6
        assert pic.decision == "KILL"
    
    def test_hold_decision(self):
        """Score 5-15 should return HOLD."""
        pic = PICScore(probability=3, impact=3, confidence=3)
        assert pic.score == 5.4
        assert pic.decision == "HOLD"
        
        pic = PICScore(probability=4, impact=4, confidence=3)
        assert pic.score == 9.6
        assert pic.decision == "HOLD"
    
    def test_publish_decision(self):
        """Score > 15 should return PUBLISH."""
        pic = PICScore(probability=5, impact=4, confidence=4)
        assert pic.score == 16.0
        assert pic.decision == "PUBLISH"
    
    def test_validation_bounds(self):
        """Values must be 1-5."""
        with pytest.raises(ValidationError):
            PICScore(probability=0, impact=3, confidence=3)
        
        with pytest.raises(ValidationError):
            PICScore(probability=6, impact=3, confidence=3)


class TestDocumentSkeleton:
    """Tests for Architect output schema."""
    
    def test_minimal_valid_skeleton(self):
        """Should accept minimal valid input."""
        skeleton = DocumentSkeleton(
            featured=FeaturedPlan(
                region="Middle East",
                headline="Tehran's New Calculus",
                angle="Iran's new calculus after Damascus",
                story_archetype="CRISIS",
                rationale="Most significant power shift in 5 years"
            ),
            sections=[
                SectionDecision(
                    region="East Asia",
                    headline="The Squeeze Continues",
                    treatment="FULL",
                    archetype="TREND",
                    word_target=300,
                    position_rationale="Second most impactful story",
                    rationale="China trade patterns shifting"
                )
            ],
            narrative_arc="American retrenchment opens doors globally",
            narrative_arc_explanation="Theme connects 4 of 5 regions",
            section_groups=[
                SectionGroup(title="The Week's Story", sections=["Middle East", "East Asia"])
            ],
            organizing_principle="THEMATIC"
        )
        assert skeleton.featured.region == "Middle East"
        assert skeleton.featured.headline == "Tehran's New Calculus"
        assert len(skeleton.sections) == 1
        assert skeleton.killed == []
        assert skeleton.quick_hits == []
    
    def test_with_killed_regions(self):
        """Should track killed regions."""
        skeleton = DocumentSkeleton(
            featured=FeaturedPlan(
                region="Europe",
                headline="Europe — Test",
                angle="Test",
                story_archetype="TREND",
                rationale="Test"
            ),
            sections=[],
            killed=[
                KilledRegion(
                    region="South Asia",
                    reason="NOTHING_NEW",
                    pic_score=3.2,
                    events_count=2,
                    highest_severity=4,
                    brief_explanation="No significant developments this week"
                )
            ],
            narrative_arc="Test arc",
            narrative_arc_explanation="Test",
            section_groups=[],
            organizing_principle="REGIONAL"
        )
        assert len(skeleton.killed) == 1
        assert skeleton.killed[0].region == "South Asia"
        assert skeleton.killed[0].events_count == 2


class TestSectionBlueprint:
    """Tests for Structure Agent output schema."""
    
    def test_complete_blueprint(self):
        """Should accept complete blueprint."""
        blueprint = SectionBlueprint(
            region="East Asia",
            archetype="CRISIS",
            word_target=300,
            hook_type="STATEMENT",
            hook_draft="The Taiwan Strait has not been this tense since 1996.",
            paragraphs=[
                ParagraphPlan(
                    purpose="What just happened",
                    word_target=60,
                    key_facts=["12 PLAAF sorties", "Median line crossings"],
                    beat="What just happened"
                ),
                ParagraphPlan(
                    purpose="Why it matters NOW",
                    word_target=80,
                    key_facts=["US carrier group repositioned"],
                    beat="Why it matters NOW"
                )
            ],
            key_events=[
                EventEmphasis(
                    event_id="evt_123",
                    emphasis="HIGH",
                    role="Trigger event"
                )
            ],
            closing_type="WARNING",
            closing_draft="The next 72 hours will determine whether this is posturing or prelude."
        )
        assert blueprint.region == "East Asia"
        assert len(blueprint.paragraphs) == 2
        assert blueprint.paragraphs[0].word_target == 60


class TestCoVeResult:
    """Tests for Chain of Verification schema."""
    
    def test_hallucination_rate_empty(self):
        """Empty results should have 0 rate."""
        cove = CoVeResult()
        assert cove.hallucination_rate == 0.0
        assert cove.has_critical_hallucinations is False
    
    def test_hallucination_rate_calculation(self):
        """Should calculate rate correctly."""
        cove = CoVeResult(
            claims_extracted=[
                FactualClaim(
                    id="c1", claim_text="Test", claim_type="STATISTIC",
                    source_sentence="Test sentence"
                )
            ],
            verification_results=[
                VerificationResult(
                    claim_id="c1",
                    verification_question="Is X true?",
                    independent_answer="No",
                    draft_answer="Yes",
                    match=False
                ),
                VerificationResult(
                    claim_id="c2",
                    verification_question="Is Y true?",
                    independent_answer="Yes",
                    draft_answer="Yes",
                    match=True
                )
            ]
        )
        assert cove.hallucination_rate == 0.5
    
    def test_critical_hallucination_flag(self):
        """Should detect critical hallucinations."""
        cove = CoVeResult(
            hallucination_flags=[
                HallucinationFlag(
                    claim_id="c1",
                    severity="CRITICAL",
                    explanation="Wrong date"
                )
            ]
        )
        assert cove.has_critical_hallucinations is True
        
        cove = CoVeResult(
            hallucination_flags=[
                HallucinationFlag(
                    claim_id="c1",
                    severity="MINOR",
                    explanation="Slight embellishment"
                )
            ]
        )
        assert cove.has_critical_hallucinations is False


class TestCriticResults:
    """Tests for ContentCriticResult and StyleCriticResult schemas."""
    
    def test_content_critic_result(self):
        """ContentCriticResult should validate correctly."""
        result = ContentCriticResult(
            passed=True,
            score=85,
            angle_delivery_score=15,
            prediction_score=15,
            constraints_score=15,
            density_score=10,
            sherman_kent_score=10,
            coverage_score=10,
            sacred_elements_score=10,
            issues=[],
            feedback="",
        )
        assert result.passed is True
        assert result.score == 85
    
    def test_style_critic_result(self):
        """StyleCriticResult should validate correctly."""
        result = StyleCriticResult(
            passed=False,
            score=65,
            weasel_free_score=15,
            sentence_variety_score=10,
            active_voice_score=10,
            opening_punch_score=10,
            closing_resonance_score=5,
            quotable_line_score=5,
            orwell_score=10,
            fact_preservation_score=10,
            content_regression_penalty=-10,
            issues=["Too many passive voice constructions"],
            feedback="Reduce passive voice, add more punch to opening",
        )
        assert result.passed is False
        assert result.feedback is not None
        assert result.content_regression_penalty == -10


class TestPipelineState:
    """Tests for the main pipeline state."""
    
    def test_initialization(self):
        """Should initialize with defaults."""
        state = PipelineState(
            run_id="test_run_001",
            started_at=datetime.now()
        )
        assert state.run_id == "test_run_001"
        assert state.analyst_outputs == {}
        assert state.document_skeleton is None
        assert state.publishable_regions == []
        assert state.killed_regions == []
    
    def test_checkpoint_tracking(self):
        """Should track checkpoints."""
        state = PipelineState(
            run_id="test_run_002",
            started_at=datetime.now()
        )
        state.add_checkpoint("analysts_complete")
        state.add_checkpoint("architect_complete")
        
        assert len(state.checkpoints) == 2
        assert "analysts_complete" in state.checkpoints[0]
    
    def test_error_tracking(self):
        """Should track errors."""
        state = PipelineState(
            run_id="test_run_003",
            started_at=datetime.now()
        )
        state.add_error("Analyst failed for region X")
        
        assert len(state.errors) == 1
        assert "Analyst failed" in state.errors[0]


class TestExampleBank:
    """Tests for the example bank."""
    
    def test_examples_load(self):
        """Should load curated examples."""
        assert len(ECONOMIST_EXAMPLES) >= 3
    
    def test_get_style_examples_returns_string(self):
        """Should return formatted string."""
        result = get_style_examples(topic="Conflict", tone="Urgent", limit=2)
        assert isinstance(result, str)
        assert "Example" in result
        assert "Before" in result
        assert "After" in result
    
    def test_topic_filtering(self):
        """Should filter by topic."""
        result = get_style_examples(topic="Elections", limit=5)
        assert "Elections" in result


class TestNegativeConstraints:
    """Tests for cliché detection and brevity substitutions."""
    
    def test_cliche_detection(self):
        """Should detect clichés."""
        text = "At the end of the day, it remains to be seen whether this will be a game changer."
        cliches = check_for_cliches(text)
        assert "at the end of the day" in cliches
        assert "remains to be seen" in cliches
        assert "game changer" in cliches
    
    def test_no_cliches(self):
        """Clean text should return empty list."""
        text = "The election produced unexpected results in three key districts."
        cliches = check_for_cliches(text)
        assert cliches == []
    
    def test_brevity_substitutions(self):
        """Should substitute long words for short ones."""
        text = "We will utilize this methodology to facilitate the implementation."
        result = apply_brevity_substitutions(text)
        assert "use" in result
        assert "method" in result
        assert "help" in result
        # Original words should be gone
        assert "utilize" not in result
        assert "methodology" not in result
        assert "facilitate" not in result


class TestSacredElements:
    """Tests for entity preservation schema."""
    
    def test_sacred_elements_creation(self):
        """Should store all element types."""
        sacred = SacredElements(
            proper_nouns=["Vladimir Putin", "Gazprom"],
            statistics=["$420 billion", "23%"],
            dates=["March 15, 2026"],
            quotes=["We will not tolerate interference."],
            event_ids=["evt_001", "evt_002"]
        )
        assert len(sacred.proper_nouns) == 2
        assert sacred.statistics[0] == "$420 billion"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
