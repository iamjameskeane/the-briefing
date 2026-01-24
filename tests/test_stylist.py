"""
Tests for the Stylist Agent.

Validates:
- Orwell filter application
- Sacred elements verification
- Sacred elements extraction
- Cliché detection
"""

import pytest

from agents.stylist import (
    StylistInput,
    apply_orwell_filter,
    verify_sacred_elements,
    extract_sacred_elements,
)
from agents.schemas import (
    AnalystOutput,
    ActorAnalysis,
    ConstraintSet,
    FuturesWheel,
    CompetingHypothesis,
)
from state import SacredElements, SectionBlueprint, ParagraphPlan, EventEmphasis


def create_mock_analyst_output(region: str) -> AnalystOutput:
    """Create a mock AnalystOutput for testing."""
    return AnalystOutput(
        region=region,
        geopolitical_archetype="Security Dilemma",
        archetype_explanation="Test explanation",
        primary_actors=[
            ActorAnalysis(
                actor="Vladimir Putin",
                intent="Maintain strategic depth",
                constraints=ConstraintSet(
                    geographic="Test geo",
                    economic="Test econ",
                    political="Test pol"
                ),
                likely_action="Test action"
            ),
            ActorAnalysis(
                actor="Beijing",
                intent="Economic expansion",
                constraints=ConstraintSet(
                    geographic="Pacific access",
                    economic="Trade dependencies",
                    political="Party stability"
                ),
                likely_action="Continue investment"
            )
        ],
        futures_wheel=FuturesWheel(
            driver_event="Major summit announced",
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


def create_mock_blueprint() -> SectionBlueprint:
    """Create a mock SectionBlueprint for testing."""
    return SectionBlueprint(
        region="Middle East",
        archetype="CRISIS",
        word_target=300,
        hook_type="STATEMENT",
        hook_draft="Test hook",
        paragraphs=[
            ParagraphPlan(
                purpose="What happened",
                word_target=100,
                key_facts=[],
                beat="What happened"
            )
        ],
        key_events=[
            EventEmphasis(
                event_id="evt_001",
                emphasis="HIGH",
                role="Driver"
            )
        ],
        closing_type="PREDICTION",
        closing_draft="Test closing"
    )


class TestOrwellFilter:
    """Tests for Orwell filter post-processing."""
    
    def test_brevity_substitutions(self):
        """Should replace long words with short ones."""
        text = "We must utilize this methodology to facilitate the implementation."
        result = apply_orwell_filter(text)
        
        assert "use" in result
        assert "method" in result
        assert "help" in result
        # Original words should be gone
        assert "utilize" not in result
        assert "methodology" not in result
        assert "facilitate" not in result
    
    def test_cliche_removal(self):
        """Should remove or replace clichés."""
        text = "At the end of the day, this is a perfect storm that will be a game changer."
        result = apply_orwell_filter(text)
        
        assert "at the end of the day" not in result.lower()
        assert "perfect storm" not in result.lower()
        assert "game changer" not in result.lower()
    
    def test_double_space_cleanup(self):
        """Should clean up double spaces."""
        text = "This  has  multiple  double  spaces."
        result = apply_orwell_filter(text)
        
        assert "  " not in result
    
    def test_preserves_good_prose(self):
        """Should not damage already-good prose."""
        text = "Moscow demands compliance. Beijing watches carefully."
        result = apply_orwell_filter(text)
        
        # Should be largely unchanged
        assert "Moscow demands compliance" in result
        assert "Beijing watches carefully" in result


class TestSacredElementsVerification:
    """Tests for sacred elements preservation checking."""
    
    def test_proper_noun_preserved(self):
        """Should detect when proper nouns are preserved."""
        sacred = SacredElements(
            proper_nouns=["Vladimir Putin", "Gazprom"],
            statistics=[],
            dates=[],
            quotes=[],
            event_ids=[]
        )
        original = "Vladimir Putin announced that Gazprom would increase exports."
        styled = "Vladimir Putin announced that Gazprom would increase exports."
        
        violations = verify_sacred_elements(styled, sacred, original)
        assert violations == []
    
    def test_proper_noun_missing(self):
        """Should detect when proper nouns are missing."""
        sacred = SacredElements(
            proper_nouns=["Vladimir Putin", "Gazprom"],
            statistics=[],
            dates=[],
            quotes=[],
            event_ids=[]
        )
        original = "Vladimir Putin announced that Gazprom would increase exports."
        styled = "The Russian president announced that the energy company would increase exports."
        
        violations = verify_sacred_elements(styled, sacred, original)
        assert len(violations) == 2
        assert any("Vladimir Putin" in v for v in violations)
        assert any("Gazprom" in v for v in violations)
    
    def test_statistic_preserved(self):
        """Should detect when statistics are preserved."""
        sacred = SacredElements(
            proper_nouns=[],
            statistics=["$420 billion", "23%"],
            dates=[],
            quotes=[],
            event_ids=[]
        )
        original = "The deal was worth $420 billion, a 23% increase."
        styled = "The deal was worth $420 billion, a 23% increase."
        
        violations = verify_sacred_elements(styled, sacred, original)
        assert violations == []
    
    def test_statistic_missing(self):
        """Should detect when statistics are missing."""
        sacred = SacredElements(
            proper_nouns=[],
            statistics=["$420 billion"],
            dates=[],
            quotes=[],
            event_ids=[]
        )
        original = "The deal was worth $420 billion."
        styled = "The deal was worth hundreds of billions."
        
        violations = verify_sacred_elements(styled, sacred, original)
        assert len(violations) == 1
        assert "$420 billion" in violations[0]
    
    def test_date_preserved(self):
        """Should detect when dates are preserved."""
        sacred = SacredElements(
            proper_nouns=[],
            statistics=[],
            dates=["March 15, 2026"],
            quotes=[],
            event_ids=[]
        )
        original = "The summit on March 15, 2026 will be decisive."
        styled = "The summit on March 15, 2026 will be decisive."
        
        violations = verify_sacred_elements(styled, sacred, original)
        assert violations == []


class TestSacredElementsExtraction:
    """Tests for sacred elements extraction from analyst output."""
    
    def test_extracts_actor_names(self):
        """Should extract actor names as proper nouns."""
        analyst = create_mock_analyst_output("Test")
        events = []
        
        sacred = extract_sacred_elements(analyst, events)
        
        assert "Vladimir Putin" in sacred.proper_nouns
        assert "Beijing" in sacred.proper_nouns
    
    def test_extracts_driver_event_id(self):
        """Should extract driver event ID."""
        analyst = create_mock_analyst_output("Test")
        events = []
        
        sacred = extract_sacred_elements(analyst, events)
        
        assert "evt_001" in sacred.event_ids
    
    def test_extracts_statistics_from_events(self):
        """Should extract statistics from event text."""
        analyst = create_mock_analyst_output("Test")
        events = [
            {
                "id": "evt_100",
                "title": "Trade deficit reaches $50 billion",
                "summary": "The deficit grew by 15% this quarter."
            }
        ]
        
        sacred = extract_sacred_elements(analyst, events)
        
        assert any("$50 billion" in s for s in sacred.statistics)
        assert any("15%" in s for s in sacred.statistics)
    
    def test_extracts_dates_from_events(self):
        """Should extract dates from event text."""
        analyst = create_mock_analyst_output("Test")
        events = [
            {
                "id": "evt_100",
                "title": "Summit scheduled for January 15",
                "summary": "Leaders will meet on January 15, 2026."
            }
        ]
        
        sacred = extract_sacred_elements(analyst, events)
        
        assert len(sacred.dates) > 0
        assert any("January 15" in d for d in sacred.dates)


class TestStylistInput:
    """Tests for StylistInput construction."""
    
    def test_basic_construction(self):
        """Should construct with required args."""
        blueprint = create_mock_blueprint()
        sacred = SacredElements(
            proper_nouns=["Test"],
            statistics=[],
            dates=[],
            quotes=[],
            event_ids=[]
        )
        
        input_data = StylistInput(
            region="Middle East",
            writer_draft="Test draft content.",
            blueprint=blueprint,
            sacred_elements=sacred,
        )
        
        assert input_data.region == "Middle East"
        assert input_data.tone == "Analytical"  # Default
    
    def test_custom_tone(self):
        """Should accept custom tone."""
        blueprint = create_mock_blueprint()
        sacred = SacredElements(
            proper_nouns=[],
            statistics=[],
            dates=[],
            quotes=[],
            event_ids=[]
        )
        
        input_data = StylistInput(
            region="Middle East",
            writer_draft="Test draft.",
            blueprint=blueprint,
            sacred_elements=sacred,
            tone="Urgent",
        )
        
        assert input_data.tone == "Urgent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
