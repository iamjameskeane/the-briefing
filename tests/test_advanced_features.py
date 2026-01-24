"""
Tests for advanced features.

Tests cover:
1. Writer blueprint integration and Chain of Density
2. Critic advanced checks (Edit Distance, Mirror Imaging, LLM fallback)
3. Signal Corroboration Score in Architect
4. PIR/EEI and Heartland/Rimland in Analyst
5. Context compaction in State
"""

import pytest
from datetime import datetime

# =============================================================================
# WRITER TESTS
# =============================================================================


class TestWriterBlueprintIntegration:
    """Tests for Writer's SectionBlueprint integration."""
    
    def test_writer_input_accepts_blueprint(self):
        """WriterInput should accept optional SectionBlueprint."""
        from agents.schemas import WriterInput, AnalystOutput
        from state import SectionBlueprint, ParagraphPlan
        
        # Create minimal AnalystOutput
        analyst_output = AnalystOutput(
            region="Middle East",
            geopolitical_archetype="Resource Curse",
            archetype_explanation="Oil dependence drives volatility",
            primary_actors=[],
            futures_wheel={
                "driver_event": "Oil price shock",
                "driver_event_id": "e1",
                "first_order": "Budget crisis",
                "second_order": "Social unrest",
                "third_order": "Regime instability",
            },
            competing_hypotheses={
                "consensus": "Prices will recover",
                "contrarian": "Structural decline",
                "contradicting_evidence": ["Renewables growth"],
                "evidence_event_ids": ["e2"],
            },
            confidence="MODERATE",
            confidence_rationale="Mixed signals",
        )
        
        blueprint = SectionBlueprint(
            region="Middle East",
            archetype="CRISIS",
            word_target=300,
            hook_type="STATEMENT",
            hook_draft="Oil markets signal deeper trouble.",
            paragraphs=[
                ParagraphPlan(
                    beat="What just happened",
                    purpose="Context",
                    word_target=100,
                    key_facts=["Oil price drop", "OPEC response"],
                )
            ],
            closing_type="PREDICTION",
            closing_draft="Expect further volatility.",
        )
        
        # Should not raise
        writer_input = WriterInput(
            analyst_output=analyst_output,
            section_type="regional",
            word_limit=300,
            events_for_linking=[],
            section_blueprint=blueprint,
            apply_chain_of_density=True,
        )
        
        assert writer_input.section_blueprint is not None
        assert writer_input.apply_chain_of_density is True
    
    def test_format_blueprint_structure(self):
        """Blueprint should be formatted into a structure template."""
        from agents.writer import _format_blueprint_structure
        from state import SectionBlueprint, ParagraphPlan, EventEmphasis
        
        blueprint = SectionBlueprint(
            region="Europe",
            archetype="TREND",
            word_target=250,
            hook_type="CONTRAST",
            hook_draft="Energy security has replaced climate as Brussels' priority.",
            paragraphs=[
                ParagraphPlan(
                    beat="The shift",
                    purpose="The shift",
                    word_target=80,
                    key_facts=["Policy reversal", "EU summit outcome"],
                ),
                ParagraphPlan(
                    beat="Evidence",
                    purpose="Evidence",
                    word_target=80,
                    key_facts=["Gas imports up 15%", "LNG terminals"],
                ),
            ],
            key_events=[
                EventEmphasis(
                    event_id="e1",
                    event_title="Energy Summit",
                    emphasis="HIGH",
                    role="hook",
                    how_to_use="Lead with this",
                )
            ],
            closing_type="CALLBACK",
            closing_draft="The green dream is on pause.",
        )
        
        result = _format_blueprint_structure(blueprint)
        
        # Check that key content is present
        assert "Europe" in result
        assert "250" in result  # Word target
        assert "Energy security" in result  # Hook draft
        assert "The shift" in result  # Paragraph purpose
        assert "Evidence" in result  # Paragraph purpose
        
        # Check that archetype/types are translated to guidance, not exposed as labels
        # TREND should become "Write analytically" guidance
        assert any(word in result.lower() for word in ["analytically", "pattern", "tone"])
        
        # Hook type (CONTRAST) should be translated to instruction
        assert any(word in result.lower() for word in ["contrast", "open"])
        
        # Closing type (CALLBACK) should be translated to instruction  
        assert any(word in result.lower() for word in ["callback", "echo", "end"])


# =============================================================================
# CRITIC ADVANCED TESTS
# =============================================================================


@pytest.mark.skip(reason="Edit distance check removed in favor of CoVe approach")
class TestCriticEditDistance:
    """Tests for Edit Distance check."""
    
    def test_edit_distance_clean_quote(self):
        """Should pass when quotes match sources."""
        pass
    
    def test_edit_distance_distorted_quote(self):
        """Should flag potentially distorted quotes."""
        pass


@pytest.mark.skip(reason="Mirror imaging check removed in favor of analyst coverage checks")
class TestCriticMirrorImaging:
    """Tests for Mirror Imaging check."""
    
    def test_mirror_imaging_clean(self):
        """Should pass when no mirror imaging detected."""
        pass
    
    def test_mirror_imaging_flagged(self):
        """Should flag mirror imaging language."""
        pass


# =============================================================================
# ARCHITECT TESTS
# =============================================================================


class TestSignalCorroborationScore:
    """Tests for Signal Corroboration Score."""
    
    def test_scs_noise_tier(self):
        """Single/no source should be noise tier."""
        from agents.editorial_utils import calculate_signal_corroboration_score
        
        events = [{"source": "Unknown", "title": "Rumor", "severity": 3}]
        
        score = calculate_signal_corroboration_score(events, region="Test")
        
        assert score <= 2  # Noise or single source
    
    def test_scs_candidate_tier(self):
        """Multiple diverse sources should be candidate tier."""
        from agents.editorial_utils import calculate_signal_corroboration_score
        
        events = [
            {"source": "Reuters", "title": "GDP up 5%", "severity": 8},
            {"source": "AP", "title": "Economic growth", "severity": 7},
            {"source": "Local News", "title": "Economy improving", "severity": 6},
            {"source": "Government Ministry", "title": "Official statement", "severity": 9},
        ]
        
        score = calculate_signal_corroboration_score(events, region="Test")
        
        assert score >= 4  # Candidate or critical tier
    
    def test_scs_with_source_diversity(self):
        """Should use provided source diversity if available."""
        from agents.editorial_utils import calculate_signal_corroboration_score
        
        events = [{"source": "Reuters", "title": "Event", "severity": 5}]
        source_diversity = {"Test": 5}  # Override with 5 distinct sources
        
        score = calculate_signal_corroboration_score(events, source_diversity, region="Test")
        
        assert score >= 3  # Should be elevated due to diversity


# =============================================================================
# ANALYST TESTS
# =============================================================================


class TestGeospatialClassification:
    """Tests for Heartland/Rimland classification."""
    
    def test_heartland_classification(self):
        """Heartland regions should be classified correctly."""
        from agents.analyst import get_geospatial_classification
        
        assert get_geospatial_classification("Russia") == "HEARTLAND"
        assert get_geospatial_classification("Central Asia") == "HEARTLAND"
    
    def test_rimland_classification(self):
        """Rimland regions should be classified correctly."""
        from agents.analyst import get_geospatial_classification
        
        assert get_geospatial_classification("Europe") == "RIMLAND"
        assert get_geospatial_classification("Middle East") == "RIMLAND"
        assert get_geospatial_classification("Southeast Asia") == "RIMLAND"
    
    def test_offshore_balancer_classification(self):
        """Offshore balancers should be classified correctly."""
        from agents.analyst import get_geospatial_classification
        
        assert get_geospatial_classification("United States") == "OFFSHORE_BALANCER"
        assert get_geospatial_classification("United Kingdom") == "OFFSHORE_BALANCER"
        assert get_geospatial_classification("Japan") == "RIMLAND"  # Japan is in RIMLAND list
    
    def test_other_classification(self):
        """Unknown regions should be classified as OTHER."""
        from agents.analyst import get_geospatial_classification
        
        assert get_geospatial_classification("Antarctica") == "OTHER"
        assert get_geospatial_classification("Unknown Region") == "OTHER"


# =============================================================================
# STATE TESTS
# =============================================================================


class TestContextCompaction:
    """Tests for context compaction in PipelineState."""
    
    def test_compacted_context_structure(self):
        """Compacted context should have expected structure."""
        from state import PipelineState, RegionCluster, SacredElements
        from agents.schemas import AnalystOutput
        
        state = PipelineState(
            run_id="test-123",
            started_at=datetime.now(),
        )
        
        # Add analyst output
        state.analyst_outputs["Middle East"] = AnalystOutput(
            region="Middle East",
            geopolitical_archetype="Resource Curse",
            archetype_explanation="Oil dependence",
            primary_actors=[{
                "actor": "Saudi Arabia",
                "intent": "Diversify economy",
                "constraints": {
                    "geographic": "Desert terrain limits agriculture",
                    "economic": "Oil revenue dependent",
                    "political": "Monarchy faces reform pressure",
                },
                "likely_action": "Continue Vision 2030",
            }],
            futures_wheel={
                "driver_event": "Oil price",
                "driver_event_id": "e1",
                "first_order": "Budget impact",
                "second_order": "Reform pace",
                "third_order": "Geopolitical alignment",
            },
            competing_hypotheses={
                "consensus": "Gradual reform",
                "contrarian": "Stagnation",
                "contradicting_evidence": ["Youth unemployment"],
                "evidence_event_ids": ["e2"],
            },
            confidence="MODERATE",
            confidence_rationale="Mixed indicators",
        )
        
        # Add event cluster
        state.event_clusters["Middle East"] = RegionCluster(
            region="Middle East",
            events=[{"title": "Oil Summit", "severity": 8}],
            storylines=["Energy transition"],
        )
        
        # Add sacred elements
        state.sacred_elements["Middle East"] = SacredElements(
            proper_nouns=["Saudi Arabia", "Crown Prince"],
            statistics=["$100 billion"],
            dates=["2030"],
        )
        
        context = state.get_compacted_context_for_region("Middle East")
        
        assert context["region"] == "Middle East"
        assert "analyst_summary" in context
        assert context["analyst_summary"]["archetype"] == "Resource Curse"
        assert "top_events" in context
        assert "sacred_elements" in context
    
    def test_state_summary(self):
        """State summary should be human-readable."""
        from state import PipelineState
        
        state = PipelineState(
            run_id="test-456",
            started_at=datetime.now(),
        )
        state.add_checkpoint("started")
        state.add_error("Test error")
        
        summary = state.get_state_summary()
        
        assert "test-456" in summary
        assert "Checkpoints: 1" in summary
        assert "Errors: 1" in summary
        assert "Test error" in summary


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestFullResearchImplementation:
    """High-level tests verifying all research findings are implemented."""
    
    def test_analyst_has_all_frameworks(self):
        """Analyst prompt should include all research frameworks."""
        from agents.analyst import ANALYST_SYSTEM_PROMPT
        
        prompt_lower = ANALYST_SYSTEM_PROMPT.lower()
        
        # Original research frameworks (case-insensitive)
        assert "const-o-t" in prompt_lower or "constraints-of-thought" in prompt_lower
        assert "futures wheel" in prompt_lower or "causal mapping" in prompt_lower
        assert "competing hypotheses" in prompt_lower or "ach" in prompt_lower
        assert "pmesii" in prompt_lower
        
        # additions
        assert "pir" in prompt_lower or "priority intelligence" in prompt_lower
        assert "heartland" in prompt_lower or "mackinder" in prompt_lower
    
    def test_architect_has_editorial_philosophy(self):
        """Editor prompt should include editorial philosophy (nub, PIC, etc.)."""
        from agents.editor import EDITOR_SYSTEM_PROMPT
        
        prompt_lower = EDITOR_SYSTEM_PROMPT.lower()
        
        assert "pic" in prompt_lower or "probability" in prompt_lower
        assert "delta test" in prompt_lower
        
        # Architect is now just for JSON structuring
        from agents.architect import ARCHITECT_SYSTEM_PROMPT
        arch_lower = ARCHITECT_SYSTEM_PROMPT.lower()
        assert "json" in arch_lower
    
    def test_structure_has_all_archetypes(self):
        """Structure should have all 7 archetypes."""
        from agents.structure import ARCHETYPE_TEMPLATES
        
        required = ["CRISIS", "TREND", "PIVOT", "SLEEPER", "COMPETITION", "CONSTRAINT", "LEADER"]
        for archetype in required:
            assert archetype in ARCHETYPE_TEMPLATES
    
    def test_stylist_has_orwell_filter(self):
        """Stylist should have Orwell Filter implementation."""
        from agents.stylist import STYLIST_SYSTEM_PROMPT
        
        prompt_lower = STYLIST_SYSTEM_PROMPT.lower()
        
        assert "orwell" in prompt_lower
        assert "passive" in prompt_lower
        assert "cliché" in prompt_lower or "cliche" in prompt_lower
    
    def test_critic_has_cove(self):
        """Critic should have Chain of Verification."""
        from agents.critic import run_cove
        
        # Should be callable
        assert callable(run_cove)
    
    def test_writer_has_chain_of_density(self):
        """Writer should have Chain of Density."""
        from agents.writer import _apply_chain_of_density
        
        # Should be callable
        assert callable(_apply_chain_of_density)
