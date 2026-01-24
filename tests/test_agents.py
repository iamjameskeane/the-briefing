"""
Unit tests for The Briefing Multi-Agent Pipeline.

These tests validate the agent schemas and local logic without making API calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.schemas import (
    AnalystInput,
    AnalystOutput,
    ActorAnalysis,
    ConstraintSet,
    FuturesWheel,
    CompetingHypothesis,
    WriterInput,
    SectionOutput,
    RegionContext,
)
from agents.critic import BANNED_PHRASES, SHERMAN_KENT_TERMS
from state import ContentCriticResult, StyleCriticResult
# Removed: constraints.py was deleted as unused
# Removed: _do_local_checks function no longer exists


# =============================================================================
# SCHEMA TESTS
# =============================================================================


class TestAnalystSchemas:
    """Test Analyst Agent schemas."""

    def test_constraint_set_creation(self):
        """Test ConstraintSet can be created with required fields."""
        constraints = ConstraintSet(
            geographic="Landlocked, no sea access",
            economic="Dependent on energy imports",
            political="Coalition government, weak mandate",
        )
        assert constraints.geographic == "Landlocked, no sea access"
        assert constraints.economic == "Dependent on energy imports"
        assert constraints.political == "Coalition government, weak mandate"

    def test_actor_analysis_creation(self):
        """Test ActorAnalysis can be created."""
        actor = ActorAnalysis(
            actor="Washington",
            intent="Maintain regional hegemony",
            constraints=ConstraintSet(
                geographic="Separated by ocean",
                economic="Largest economy but debt constraints",
                political="Divided Congress",
            ),
            likely_action="HIGHLY LIKELY to pursue diplomatic pressure over military",
        )
        assert actor.actor == "Washington"
        assert "hegemony" in actor.intent

    def test_futures_wheel_creation(self):
        """Test FuturesWheel requires all three orders."""
        wheel = FuturesWheel(
            driver_event="Tariff announcement",
            driver_event_id="evt-123",
            first_order="Immediate market volatility",
            second_order="Trading partners retaliate within 30 days",
            third_order="Global supply chain restructuring favoring regional blocs",
        )
        assert wheel.driver_event_id == "evt-123"
        assert "supply chain" in wheel.third_order

    def test_analyst_output_full(self):
        """Test complete AnalystOutput creation."""
        output = AnalystOutput(
            region="EUROPE",
            geopolitical_archetype="Security Dilemma",
            archetype_explanation="Actions to increase one's security decrease others'",
            primary_actors=[
                ActorAnalysis(
                    actor="Brussels",
                    intent="Strategic autonomy",
                    constraints=ConstraintSet(
                        geographic="Shared border with Russia",
                        economic="Energy dependency",
                        political="Unanimity requirement",
                    ),
                    likely_action="LIKELY to increase defense spending",
                )
            ],
            futures_wheel=FuturesWheel(
                driver_event="NATO summit",
                driver_event_id="evt-456",
                first_order="Increased defense commitments",
                second_order="Russia perceives encirclement",
                third_order="Permanent militarization of Baltic",
            ),
            competing_hypotheses=CompetingHypothesis(
                consensus="NATO will strengthen",
                contrarian="European fatigue leads to reduced commitment",
                contradicting_evidence=["Hungary veto threats", "German budget constraints"],
                evidence_event_ids=["evt-789", "evt-101"],
            ),
            confidence="MODERATE",
            confidence_rationale="Mixed signals from key actors",
        )
        assert output.region == "EUROPE"
        assert len(output.primary_actors) == 1
        assert output.confidence == "MODERATE"


class TestWriterSchemas:
    """Test Writer Agent schemas."""

    def test_writer_input_creation(self):
        """Test WriterInput can hold analyst output."""
        analyst_output = AnalystOutput(
            region="MIDDLE_EAST",
            geopolitical_archetype="Resource Curse",
            archetype_explanation="Oil wealth destabilizes governance",
            primary_actors=[],
            futures_wheel=FuturesWheel(
                driver_event="Oil price spike",
                driver_event_id="evt-001",
                first_order="Budget surplus",
                second_order="Reduced reform pressure",
                third_order="Delayed diversification",
            ),
            competing_hypotheses=CompetingHypothesis(
                consensus="Status quo continues",
                contrarian="Youth unemployment forces reform",
                contradicting_evidence=["Protests in region"],
                evidence_event_ids=["evt-002"],
            ),
            confidence="HIGH",
            confidence_rationale="Consistent historical pattern",
        )

        writer_input = WriterInput(
            analyst_output=analyst_output,
            section_type="regional",
            word_limit=250,
            events_for_linking=[{"id": "evt-001", "title": "Oil spike"}],
            revision_feedback=None,
        )

        assert writer_input.word_limit == 250
        assert writer_input.section_type == "regional"


class TestCriticResultSchemas:
    """Test Critic result schemas from state.py."""

    def test_content_critic_result_creation(self):
        """Test ContentCriticResult creation."""
        from state import ContentCriticResult, CoVeResult
        
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
            cove_result=CoVeResult(),
            issues=[],
            feedback="",
        )
        assert result.passed is True
        assert result.score == 85

    def test_style_critic_result_creation(self):
        """Test StyleCriticResult creation."""
        from state import StyleCriticResult
        
        result = StyleCriticResult(
            passed=True,
            score=80,
            weasel_free_score=20,
            sentence_variety_score=15,
            active_voice_score=15,
            opening_punch_score=15,
            closing_resonance_score=10,
            quotable_line_score=10,
            orwell_score=15,
            fact_preservation_score=10,
            issues=[],
            feedback="",
        )
        assert result.passed is True
        assert result.score == 80


# =============================================================================
# CRITIC LOCAL CHECKS TESTS
# =============================================================================


@pytest.mark.skip(reason="Local checks removed in favor of LLM-based Critic")
class TestCriticLocalChecks:
    """Test the local (non-LLM) checks in the critic."""

    def test_weasel_detection(self):
        """Test that banned phrases are detected."""
        draft_with_weasels = """
        The situation in the region remains to be seen. Various factors
        could potentially influence the outcome. Time will tell.
        """

        # Create minimal analyst output for the check
        analyst_output = AnalystOutput(
            region="TEST",
            geopolitical_archetype="Test",
            archetype_explanation="Test",
            primary_actors=[],
            futures_wheel=FuturesWheel(
                driver_event="Test",
                driver_event_id="test-1",
                first_order="Test",
                second_order="Test",
                third_order="Test effect",
            ),
            competing_hypotheses=CompetingHypothesis(
                consensus="Test",
                contrarian="Test",
                contradicting_evidence=[],
                evidence_event_ids=[],
            ),
            confidence="LOW",
            confidence_rationale="Test",
        )

        checks = _do_local_checks(draft_with_weasels, analyst_output)

        assert len(checks["weasels_found"]) >= 3
        assert "remains to be seen" in checks["weasels_found"]
        assert "time will tell" in checks["weasels_found"]

    def test_clean_draft_no_weasels(self):
        """Test that clean draft has no weasels."""
        clean_draft = """
        Constrained by depleted foreign reserves, Ankara is HIGHLY LIKELY (80-90%)
        to delay further rate cuts, forcing a pivot toward Gulf sovereign wealth funds.
        This will deepen Turkey's strategic dependency on Riyadh.
        """

        analyst_output = AnalystOutput(
            region="MIDDLE_EAST",
            geopolitical_archetype="Test",
            archetype_explanation="Test",
            primary_actors=[],
            futures_wheel=FuturesWheel(
                driver_event="Test",
                driver_event_id="test-1",
                first_order="Test",
                second_order="Test",
                third_order="dependency on Riyadh",
            ),
            competing_hypotheses=CompetingHypothesis(
                consensus="Test",
                contrarian="Test",
                contradicting_evidence=[],
                evidence_event_ids=[],
            ),
            confidence="LOW",
            confidence_rationale="Test",
        )

        checks = _do_local_checks(clean_draft, analyst_output)

        assert len(checks["weasels_found"]) == 0
        assert "HIGHLY LIKELY" in checks["kent_terms_found"]

    def test_sherman_kent_detection(self):
        """Test Sherman Kent terms are detected."""
        draft = """
        This outcome is ALMOST CERTAIN (95%). However, a REMOTE possibility
        exists that the alternative scenario occurs.
        """

        analyst_output = AnalystOutput(
            region="TEST",
            geopolitical_archetype="Test",
            archetype_explanation="Test",
            primary_actors=[],
            futures_wheel=FuturesWheel(
                driver_event="Test",
                driver_event_id="test-1",
                first_order="Test",
                second_order="Test",
                third_order="Test",
            ),
            competing_hypotheses=CompetingHypothesis(
                consensus="Test",
                contrarian="Test",
                contradicting_evidence=[],
                evidence_event_ids=[],
            ),
            confidence="LOW",
            confidence_rationale="Test",
        )

        checks = _do_local_checks(draft, analyst_output)

        assert "ALMOST CERTAIN" in checks["kent_terms_found"]
        assert "REMOTE" in checks["kent_terms_found"]

    def test_constraint_indicator_detection(self):
        """Test constraint language is detected."""
        draft = """
        Constrained by limited resources and forced to choose between allies,
        Berlin cannot maintain its current position. The government lacks the
        political capital due to coalition dynamics.
        """

        analyst_output = AnalystOutput(
            region="EUROPE",
            geopolitical_archetype="Test",
            archetype_explanation="Test",
            primary_actors=[],
            futures_wheel=FuturesWheel(
                driver_event="Test",
                driver_event_id="test-1",
                first_order="Test",
                second_order="Test",
                third_order="Test",
            ),
            competing_hypotheses=CompetingHypothesis(
                consensus="Test",
                contrarian="Test",
                contradicting_evidence=[],
                evidence_event_ids=[],
            ),
            confidence="LOW",
            confidence_rationale="Test",
        )

        checks = _do_local_checks(draft, analyst_output)

        assert "constrained by" in checks["constraint_indicators"]
        assert "forced to" in checks["constraint_indicators"]
        assert "cannot" in checks["constraint_indicators"]


# =============================================================================
# CONSTRAINTS DATABASE TESTS
# =============================================================================


@pytest.mark.skip(reason="constraints.py deleted - unused region lookups")
class TestConstraintsDatabase:
    """Test the pre-computed region constraints."""

    def test_all_major_regions_have_context(self):
        """Test that all major regions have pre-computed context."""
        required_regions = [
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

        for region in required_regions:
            context = get_region_context(region)
            assert context is not None
            assert len(context.geography) > 50, f"{region} geography too short"
            assert len(context.economics) > 50, f"{region} economics too short"
            assert len(context.politics) > 50, f"{region} politics too short"

    def test_region_context_to_prompt_string(self):
        """Test that context can be formatted for prompts."""
        context = get_region_context("MIDDLE_EAST")
        prompt_str = context.to_prompt_string()

        assert "Geographic Constraints" in prompt_str
        assert "Economic Constraints" in prompt_str
        assert "Political Constraints" in prompt_str
        assert "Key Relationships" in prompt_str
        assert "Historical Patterns" in prompt_str

    def test_unknown_region_gets_default(self):
        """Test that unknown regions get default context."""
        context = get_region_context("UNKNOWN_REGION_XYZ")
        assert "No pre-computed" in context.geography

    def test_partial_region_match(self):
        """Test that partial region names match."""
        context = get_region_context("middle_east")
        assert len(context.geography) > 50  # Should find MIDDLE_EAST

        context2 = get_region_context("EAST-ASIA")
        assert len(context2.geography) > 50  # Should find EAST_ASIA


# =============================================================================
# SECTION OUTPUT TESTS
# =============================================================================


class TestSectionOutput:
    """Test the SectionOutput dataclass."""

    def test_section_output_creation(self):
        """Test SectionOutput can be created."""
        analyst_data = AnalystOutput(
            region="EUROPE",
            geopolitical_archetype="Balance of Power",
            archetype_explanation="States align against strongest",
            primary_actors=[],
            futures_wheel=FuturesWheel(
                driver_event="Summit",
                driver_event_id="evt-1",
                first_order="Agreement",
                second_order="Implementation",
                third_order="Structural shift",
            ),
            competing_hypotheses=CompetingHypothesis(
                consensus="Cooperation",
                contrarian="Collapse",
                contradicting_evidence=[],
                evidence_event_ids=[],
            ),
            confidence="MODERATE",
            confidence_rationale="Mixed signals",
        )

        section = SectionOutput(
            region="EUROPE",
            content="## EUROPE — STABLE\n\nContent here...",
            analyst_data=analyst_data,
            critic_score=85,
            quality_warning=False,
        )

        assert section.region == "EUROPE"
        assert section.critic_score == 85
        assert section.quality_warning is False


# =============================================================================
# BANNED PHRASES COMPREHENSIVE TEST
# =============================================================================


class TestBannedPhrases:
    """Ensure all banned phrases are properly defined."""

    def test_banned_phrases_list(self):
        """Test that banned phrases list is comprehensive."""
        assert "remains to be seen" in BANNED_PHRASES
        assert "time will tell" in BANNED_PHRASES
        assert "could potentially" in BANNED_PHRASES
        assert "significant developments" in BANNED_PHRASES
        assert "various factors" in BANNED_PHRASES

    def test_sherman_kent_terms_list(self):
        """Test that Sherman Kent terms are defined."""
        assert "ALMOST CERTAIN" in SHERMAN_KENT_TERMS
        assert "HIGHLY LIKELY" in SHERMAN_KENT_TERMS
        assert "LIKELY" in SHERMAN_KENT_TERMS
        assert "ROUGHLY EVEN" in SHERMAN_KENT_TERMS
        assert "UNLIKELY" in SHERMAN_KENT_TERMS
        assert "HIGHLY UNLIKELY" in SHERMAN_KENT_TERMS
        assert "REMOTE" in SHERMAN_KENT_TERMS
