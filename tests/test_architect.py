"""
Tests for the Architect Agent.

Validates:
- PIC scoring and decision logic
- Delta test application
- Fallback skeleton generation
- ArchitectInput construction
"""

import pytest
from datetime import datetime

from agents.editor import EditorInput
from agents.architect import ArchitectInput
from agents.editorial_utils import calculate_pic_score, apply_delta_test
from agents.schemas import (
    AnalystOutput,
    ActorAnalysis,
    ConstraintSet,
    FuturesWheel,
    CompetingHypothesis,
)
from state import PICScore


def create_mock_analyst_output(
    region: str,
    confidence: str = "MODERATE",
    third_order: str = "Regional tensions will persist",
) -> AnalystOutput:
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
            third_order=third_order
        ),
        competing_hypotheses=CompetingHypothesis(
            consensus="Main view",
            contrarian="Alternative view",
            contradicting_evidence=["Evidence 1"],
            evidence_event_ids=["evt_002"]
        ),
        pmesii_tags={"Political": ["evt_001"]},
        confidence=confidence,
        confidence_rationale="Test rationale"
    )


class TestPICScoring:
    """Tests for PIC Matrix scoring."""
    
    def test_calculate_pic_score_kill(self):
        """Low scores should result in KILL decision."""
        pic, decision = calculate_pic_score(
            probability=2,
            impact=2,
            confidence=2
        )
        assert pic.score == 1.6  # (2*2*2)/5
        assert decision == "KILL"
    
    def test_calculate_pic_score_hold(self):
        """Medium scores should result in HOLD decision."""
        pic, decision = calculate_pic_score(
            probability=3,
            impact=3,
            confidence=3
        )
        assert pic.score == 5.4  # (3*3*3)/5
        assert decision == "HOLD"
    
    def test_calculate_pic_score_publish(self):
        """High scores should result in PUBLISH decision."""
        pic, decision = calculate_pic_score(
            probability=5,
            impact=4,
            confidence=4
        )
        assert pic.score == 16.0  # (5*4*4)/5
        assert decision == "PUBLISH"
    
    def test_boundary_cases(self):
        """Test boundary between HOLD and PUBLISH."""
        # Score exactly 15 = HOLD
        pic, decision = calculate_pic_score(
            probability=5,
            impact=5,
            confidence=3
        )
        assert pic.score == 15.0
        assert decision == "HOLD"
        
        # Score just above 15 = PUBLISH
        pic, decision = calculate_pic_score(
            probability=5,
            impact=4,
            confidence=4
        )
        assert pic.score == 16.0
        assert decision == "PUBLISH"


class TestDeltaTest:
    """Tests for the Delta Test."""
    
    def test_structural_change_high_confidence(self):
        """Structural change with high confidence should publish."""
        analysis = create_mock_analyst_output(
            region="Test",
            confidence="HIGH",
            third_order="This will fundamentally shift the regional balance of power"
        )
        should_publish, reason = apply_delta_test(analysis)
        assert should_publish is True
        assert "structural" in reason.lower()
    
    def test_structural_change_low_confidence(self):
        """Structural change with low confidence should not publish."""
        analysis = create_mock_analyst_output(
            region="Test",
            confidence="LOW",
            third_order="This will fundamentally shift the regional balance of power"
        )
        should_publish, reason = apply_delta_test(analysis)
        assert should_publish is False
        assert "low confidence" in reason.lower()
    
    def test_no_structural_change(self):
        """No structural change should not publish."""
        analysis = create_mock_analyst_output(
            region="Test",
            confidence="HIGH",
            third_order="Regional tensions will continue as before"
        )
        should_publish, reason = apply_delta_test(analysis)
        assert should_publish is False
        assert "no significant change" in reason.lower()
    
    def test_structural_keywords(self):
        """Test various structural keywords."""
        keywords = ["shift", "realign", "collapse", "emerge", "unprecedented"]
        
        for kw in keywords:
            analysis = create_mock_analyst_output(
                region="Test",
                confidence="HIGH",
                third_order=f"This will {kw} the regional order"
            )
            should_publish, _ = apply_delta_test(analysis)
            assert should_publish is True, f"Keyword '{kw}' should trigger publish"


class TestArchitectInput:
    """Tests for ArchitectInput construction."""
    
    def test_basic_construction(self):
        """Should construct with minimal args."""
        from agents.editor import EditorDecisions
        
        mock_decisions = EditorDecisions(
            editorial_brief="Kill most clusters, publish Middle East and Americas as FULL.",
            conversation_history=[],
        )
        
        input_data = ArchitectInput(
            editor_decisions=mock_decisions,
            total_word_budget=2000,
        )
        
        assert input_data.editor_decisions is not None
        assert input_data.total_word_budget == 2000
    
    def test_custom_word_budget(self):
        """Should accept custom word budget."""
        from agents.editor import EditorDecisions
        
        mock_decisions = EditorDecisions(
            editorial_brief="Test",
            conversation_history=[],
        )
        
        input_data = ArchitectInput(
            editor_decisions=mock_decisions,
            total_word_budget=3000,
        )
        
        assert input_data.total_word_budget == 3000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
