"""
Tests for the Structure Agent.

Validates:
- Archetype template selection
- Fallback blueprint generation
- Beat word ratio calculations
"""

import pytest

from agents.structure import (
    StructureInput,
    create_fallback_blueprint,
    ARCHETYPE_TEMPLATES,
)
from agents.schemas import (
    AnalystOutput,
    ActorAnalysis,
    ConstraintSet,
    FuturesWheel,
    CompetingHypothesis,
)
from state import SectionDecision, SectionBlueprint


def create_mock_analyst_output(region: str) -> AnalystOutput:
    """Create a mock AnalystOutput for testing."""
    return AnalystOutput(
        region=region,
        geopolitical_archetype="Security Dilemma",
        archetype_explanation="Test explanation",
        primary_actors=[
            ActorAnalysis(
                actor="TestActor",
                intent="Test intent",
                constraints=ConstraintSet(
                    geographic="Test geo",
                    economic="Test econ",
                    political="Test pol"
                ),
                likely_action="Test action"
            )
        ],
        futures_wheel=FuturesWheel(
            driver_event="Test event",
            driver_event_id="evt_001",
            first_order="Immediate reaction",
            second_order="Regional response",
            third_order="Structural shift"
        ),
        competing_hypotheses=CompetingHypothesis(
            consensus="Main view",
            contrarian="Alternative view",
            contradicting_evidence=["Evidence 1"],
            evidence_event_ids=["evt_002"]
        ),
        pmesii_tags={"Political": ["evt_001"]},
        confidence="MODERATE",
        confidence_rationale="Test rationale"
    )


class TestArchetypeTemplates:
    """Tests for archetype template structure."""
    
    def test_all_archetypes_defined(self):
        """Should have templates for all archetypes."""
        expected = ["CRISIS", "TREND", "PIVOT", "SLEEPER", "COMPETITION", "CONSTRAINT"]
        for archetype in expected:
            assert archetype in ARCHETYPE_TEMPLATES
    
    def test_template_structure(self):
        """Each template should have required fields."""
        for name, template in ARCHETYPE_TEMPLATES.items():
            assert "description" in template, f"{name} missing description"
            assert "beats" in template, f"{name} missing beats"
            assert "hook_types" in template, f"{name} missing hook_types"
            assert "closing_types" in template, f"{name} missing closing_types"
            assert "pacing" in template, f"{name} missing pacing"
    
    def test_beat_ratios_sum_to_one(self):
        """Beat word ratios should sum to 1.0 for each archetype."""
        for name, template in ARCHETYPE_TEMPLATES.items():
            total = sum(beat["word_ratio"] for beat in template["beats"])
            assert abs(total - 1.0) < 0.01, f"{name} ratios sum to {total}, not 1.0"
    
    def test_beats_have_purpose_and_ratio(self):
        """Each beat should have purpose and word_ratio."""
        for name, template in ARCHETYPE_TEMPLATES.items():
            for i, beat in enumerate(template["beats"]):
                assert "purpose" in beat, f"{name} beat {i} missing purpose"
                assert "word_ratio" in beat, f"{name} beat {i} missing word_ratio"
                assert 0 < beat["word_ratio"] < 1, f"{name} beat {i} invalid ratio"


class TestFallbackBlueprint:
    """Tests for fallback blueprint generation."""
    
    def test_basic_fallback(self):
        """Should create valid blueprint."""
        analyst = create_mock_analyst_output("Middle East")
        decision = SectionDecision(
            region="Middle East",
            headline="Middle East — Test Headline",
            treatment="FULL",
            archetype="CRISIS",
            word_target=300,
            position_rationale="Test position",
            rationale="Test"
        )
        
        blueprint = create_fallback_blueprint("Middle East", analyst, decision)
        
        assert blueprint.region == "Middle East"
        assert blueprint.archetype == "CRISIS"
        assert blueprint.word_target == 300
        assert len(blueprint.paragraphs) > 0
    
    def test_fallback_uses_correct_archetype(self):
        """Should use archetype from section decision."""
        analyst = create_mock_analyst_output("Test")
        decision = SectionDecision(
            region="Test",
            headline="Test — Trend Headline",
            treatment="SHORT",
            archetype="TREND",
            word_target=200,
            position_rationale="Test position",
            rationale="Test"
        )
        
        blueprint = create_fallback_blueprint("Test", analyst, decision)
        
        assert blueprint.archetype == "TREND"
        # TREND template has 4 beats
        assert len(blueprint.paragraphs) == 4
    
    def test_fallback_word_allocation(self):
        """Paragraph word targets should sum to section target."""
        analyst = create_mock_analyst_output("Test")
        decision = SectionDecision(
            region="Test",
            headline="Test — Word Allocation",
            treatment="FULL",
            archetype="CRISIS",
            word_target=400,
            position_rationale="Test position",
            rationale="Test"
        )
        
        blueprint = create_fallback_blueprint("Test", analyst, decision)
        
        total_words = sum(p.word_target for p in blueprint.paragraphs)
        # Should be within 1 word due to rounding
        assert abs(total_words - 400) <= len(blueprint.paragraphs)
    
    def test_fallback_includes_driver_event(self):
        """Should include driver event as HIGH emphasis."""
        analyst = create_mock_analyst_output("Test")
        decision = SectionDecision(
            region="Test",
            headline="Test — Driver Event",
            treatment="FULL",
            archetype="CRISIS",
            word_target=300,
            position_rationale="Test position",
            rationale="Test"
        )
        
        blueprint = create_fallback_blueprint("Test", analyst, decision)
        
        assert len(blueprint.key_events) >= 1
        assert blueprint.key_events[0].emphasis == "HIGH"
        assert blueprint.key_events[0].event_id == "evt_001"
    
    def test_fallback_hook_and_closing(self):
        """Should have hook and closing drafts."""
        analyst = create_mock_analyst_output("Middle East")
        decision = SectionDecision(
            region="Middle East",
            headline="Middle East — Hook Test",
            treatment="FULL",
            archetype="CRISIS",
            word_target=300,
            position_rationale="Test position",
            rationale="Test"
        )
        
        blueprint = create_fallback_blueprint("Middle East", analyst, decision)
        
        assert blueprint.hook_draft != ""
        assert "Middle East" in blueprint.hook_draft
        assert blueprint.closing_draft != ""
        assert "Middle East" in blueprint.closing_draft
    
    def test_fallback_unknown_archetype(self):
        """Should default to TREND for unknown archetype."""
        analyst = create_mock_analyst_output("Test")
        decision = SectionDecision(
            region="Test",
            headline="Test — Unknown Archetype",
            treatment="FULL",
            archetype="UNKNOWN_TYPE",  # Not in templates
            word_target=300,
            position_rationale="Test position",
            rationale="Test"
        )
        
        blueprint = create_fallback_blueprint("Test", analyst, decision)
        
        # Should still produce valid blueprint using TREND as default
        assert blueprint.region == "Test"
        assert len(blueprint.paragraphs) > 0


class TestStructureInput:
    """Tests for StructureInput construction."""
    
    def test_basic_construction(self):
        """Should construct with required args."""
        analyst = create_mock_analyst_output("Middle East")
        decision = SectionDecision(
            region="Middle East",
            headline="Middle East — Construction Test",
            treatment="FULL",
            archetype="CRISIS",
            word_target=300,
            position_rationale="Test position",
            rationale="Test"
        )
        
        input_data = StructureInput(
            region="Middle East",
            analyst_output=analyst,
            section_decision=decision,
            events=[{"id": "evt_001", "title": "Test"}]
        )
        
        assert input_data.region == "Middle East"
        assert len(input_data.events) == 1


class TestBlueprintValidation:
    """Tests for SectionBlueprint schema."""
    
    def test_blueprint_paragraph_plan(self):
        """ParagraphPlan should have all required fields."""
        from state import ParagraphPlan
        
        plan = ParagraphPlan(
            purpose="Test purpose",
            word_target=100,
            key_facts=["Fact 1", "Fact 2"],
            beat="Test beat"
        )
        
        assert plan.purpose == "Test purpose"
        assert plan.word_target == 100
        assert len(plan.key_facts) == 2
    
    def test_blueprint_event_emphasis(self):
        """EventEmphasis should validate emphasis levels."""
        from state import EventEmphasis
        
        emphasis = EventEmphasis(
            event_id="evt_001",
            emphasis="HIGH",
            role="Primary driver"
        )
        
        assert emphasis.emphasis == "HIGH"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
