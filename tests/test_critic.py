"""
Tests for the Critic Agent.

Validates:
- Content Critic (for Writer loop)
- Style Critic (for Stylist loop)
- Chain of Verification (CoVe)
- Content preservation checks
"""

import pytest

from agents.critic import (
    run_content_critic,
    run_style_critic,
    run_cove,
    extract_factual_claims,
    BANNED_PHRASES,
    SHERMAN_KENT_TERMS,
)
from agents.schemas import (
    AnalystOutput,
    ActorAnalysis,
    ConstraintSet,
    FuturesWheel,
    CompetingHypothesis,
)
from state import SacredElements, ContentCriticResult, StyleCriticResult


def create_mock_analyst_output(region: str) -> AnalystOutput:
    """Create a mock AnalystOutput for testing."""
    return AnalystOutput(
        region=region,
        geopolitical_archetype="Security Dilemma",
        archetype_explanation="Test explanation",
        primary_actors=[
            ActorAnalysis(
                actor="Moscow",
                intent="Strategic depth",
                constraints=ConstraintSet(
                    geographic="North European Plain",
                    economic="Sanctions pressure",
                    political="Domestic stability"
                ),
                likely_action="Continue defensive posture"
            )
        ],
        futures_wheel=FuturesWheel(
            driver_event="Summit announcement",
            driver_event_id="evt_001",
            first_order="Immediate reaction",
            second_order="Regional realignment",
            third_order="Structural shift in alliance patterns"
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


class TestContentCritic:
    """Tests for ContentCritic (Writer evaluation)."""
    
    def test_good_content_passes(self):
        """Well-written content should pass."""
        draft = """Moscow is constrained by depleted reserves and limited by Western sanctions. 
        The Kremlin cannot sustain current spending levels. This HIGHLY LIKELY will force a pivot 
        toward China, fundamentally shifting the alliance patterns in Eurasia. $50 billion in 
        reserves remain, down 30% from January 2025. Vladimir Putin faces a structural dilemma."""
        
        analyst = create_mock_analyst_output("Russia")
        sacred = SacredElements(proper_nouns=["Moscow"], statistics=[], dates=[], quotes=[], event_ids=[])
        events = [{"id": "evt_001", "title": "Test event", "summary": "$50 billion reserves"}]
        
        result = run_content_critic(draft, analyst, sacred, events, region="Russia")
        
        assert isinstance(result, ContentCriticResult)
        assert result.passed is True
        assert result.score >= 75  # Content pass threshold
        assert len(result.issues) <= 2  # Few or no issues
    
    def test_missing_prediction_flagged(self):
        """Content without prediction should be flagged."""
        draft = """Moscow held a summit. Various officials attended. 
        The situation was discussed. Many topics were covered."""
        
        analyst = create_mock_analyst_output("Russia")
        sacred = SacredElements(proper_nouns=[], statistics=[], dates=[], quotes=[], event_ids=[])
        events = []
        
        result = run_content_critic(draft, analyst, sacred, events, region="Russia")
        
        assert any("prediction" in i.lower() for i in result.issues)
    
    def test_missing_constraints_flagged(self):
        """Content without constraints should be flagged."""
        draft = """Moscow will expand its influence. The trajectory is clear.
        Expect significant developments in the coming months. LIKELY outcomes include..."""
        
        analyst = create_mock_analyst_output("Russia")
        sacred = SacredElements(proper_nouns=[], statistics=[], dates=[], quotes=[], event_ids=[])
        events = []
        
        result = run_content_critic(draft, analyst, sacred, events, region="Russia")
        
        assert any("constraint" in i.lower() for i in result.issues)
    
    def test_low_density_flagged(self):
        """Low entity density should be flagged."""
        draft = """The situation is developing. Things are changing. 
        Many factors are at play. The outcome remains uncertain."""
        
        analyst = create_mock_analyst_output("Russia")
        sacred = SacredElements(proper_nouns=[], statistics=[], dates=[], quotes=[], event_ids=[])
        events = []
        
        result = run_content_critic(draft, analyst, sacred, events, region="Russia")
        
        assert any("density" in i.lower() for i in result.issues)
    
    def test_sherman_kent_detection(self):
        """Should detect Sherman Kent terms."""
        draft = """Moscow is HIGHLY LIKELY to pivot toward Beijing. The constraints
        force this outcome. We expect significant changes by March 2026."""
        
        analyst = create_mock_analyst_output("Russia")
        sacred = SacredElements(proper_nouns=[], statistics=[], dates=[], quotes=[], event_ids=[])
        events = []
        
        result = run_content_critic(draft, analyst, sacred, events, region="Russia")
        
        # Should not have Sherman Kent issue (term is present)
        assert not any("sherman kent" in i.lower() for i in result.issues)


class TestStyleCritic:
    """Tests for StyleCritic (Stylist evaluation)."""
    
    def test_good_style_passes(self):
        """Good style should pass."""
        # Writer draft (source of truth)
        writer_draft = """Moscow has a problem. Reserves are depleting at $5 billion per month.
        Beijing watches carefully. The Kremlin's options narrow each day.
        Sanctions force uncomfortable choices. Expect a pivot by summer."""
        
        # Styled draft (same content, maybe slightly improved)
        styled_draft = writer_draft  # For this test, same as writer
        
        sacred = SacredElements(proper_nouns=["Moscow"], statistics=["$5 billion"], dates=[], quotes=[], event_ids=[])
        
        result = run_style_critic(styled_draft, writer_draft, sacred, region="Russia")
        
        assert isinstance(result, StyleCriticResult)
        assert result.passed is True
        assert result.score >= 70  # Style pass threshold
    
    def test_weasel_phrases_flagged(self):
        """Banned phrases should be flagged."""
        writer_draft = """Moscow is constrained by sanctions. HIGHLY LIKELY to pivot."""
        styled_draft = """It remains to be seen what Moscow will do. At the end of the day,
        this could potentially be a game changer. Time will tell."""
        
        sacred = SacredElements(proper_nouns=[], statistics=[], dates=[], quotes=[], event_ids=[])
        
        result = run_style_critic(styled_draft, writer_draft, sacred, region="Russia")
        
        assert result.passed is False
        assert any("banned" in i.lower() or "remove" in i.lower() for i in result.issues)
    
    def test_passive_voice_detection(self):
        """Passive voice indicator counting should work."""
        writer_draft = """Moscow decided. Sanctions hit. Results followed."""
        styled_draft = """Progress was made. The decision was made. The treaty was signed.
        Terms were discussed. An agreement was reached. Things were done.
        Nothing was achieved. Everything was lost."""
        
        sacred = SacredElements(proper_nouns=[], statistics=[], dates=[], quotes=[], event_ids=[])
        
        result = run_style_critic(styled_draft, writer_draft, sacred, region="Russia")
        
        # Function should run without error
        assert isinstance(result, StyleCriticResult)
        assert isinstance(result.score, int)
        assert isinstance(result.passed, bool)
        assert isinstance(result.issues, list)
    
    def test_sentence_variety_checked(self):
        """Lack of sentence variety should be flagged."""
        writer_draft = """Moscow announced policies. Washington responded. Tensions escalated."""
        styled_draft = """The Kremlin announced new policies. The Pentagon responded quickly.
        The situation escalated rapidly. The markets reacted negatively.
        The alliance strengthened further. The trajectory became clear."""
        
        sacred = SacredElements(proper_nouns=[], statistics=[], dates=[], quotes=[], event_ids=[])
        
        result = run_style_critic(styled_draft, writer_draft, sacred, region="Russia")
        
        # All sentences start with "The" - should flag variety or opening
        assert any("variety" in i.lower() or "opening" in i.lower() or "repetitive" in i.lower() for i in result.issues)
    
    def test_strong_opening_detected(self):
        """Strong openings should score well."""
        writer_draft = """Moscow faces a reckoning. Options narrow."""
        styled_draft = """Moscow faces a reckoning. Reserves depleted, sanctions biting, 
        Beijing circling—the Kremlin's options narrow daily."""
        
        sacred = SacredElements(proper_nouns=["Moscow"], statistics=[], dates=[], quotes=[], event_ids=[])
        
        result = run_style_critic(styled_draft, writer_draft, sacred, region="Russia")
        
        # Should not flag opening (starts with "Moscow", not "The")
        assert not any("opening" in i.lower() and "strengthen" in i.lower() for i in result.issues)
    
    def test_content_regression_flagged(self):
        """Stylist that removes analytical content should be penalized."""
        # Writer draft with predictions, constraints, Sherman Kent
        writer_draft = """Moscow is constrained by depleted reserves. The Kremlin cannot sustain
        current spending. This HIGHLY LIKELY will force a pivot. $50 billion remains."""
        
        # Styled draft that removes analysis
        styled_draft = """Moscow's situation is difficult. The Kremlin faces challenges.
        Changes may occur. The situation continues."""
        
        sacred = SacredElements(proper_nouns=["Moscow"], statistics=["$50 billion"], dates=[], quotes=[], event_ids=[])
        
        result = run_style_critic(styled_draft, writer_draft, sacred, region="Russia")
        
        # Should fail due to content regression
        assert result.passed is False
        assert any("regression" in i.lower() or "lost" in i.lower() for i in result.issues)


class TestCoVe:
    """Tests for Chain of Verification."""
    
    def test_extract_statistics(self):
        """Should extract statistics from draft."""
        draft = """The deficit reached $50 billion, up 15% from last year. 
        Reserves stand at 127 million barrels."""
        
        claims = extract_factual_claims(draft)
        
        stat_claims = [c for c in claims if c.claim_type == "STATISTIC"]
        assert len(stat_claims) >= 2
        assert any("50" in c.claim_text for c in stat_claims)
        assert any("15" in c.claim_text for c in stat_claims)
    
    def test_extract_dates(self):
        """Should extract dates from draft."""
        draft = """The summit on January 15, 2026 will be decisive. 
        Negotiations began on March 3."""
        
        claims = extract_factual_claims(draft)
        
        date_claims = [c for c in claims if c.claim_type == "DATE"]
        assert len(date_claims) >= 1
        assert any("January" in c.claim_text for c in date_claims)
    
    def test_verify_claims_found_in_source(self):
        """Should verify claims that appear in sources."""
        draft = """The deal was worth $50 billion."""
        
        sources = [
            {"title": "Major deal announced", "summary": "The agreement totals $50 billion over five years."}
        ]
        
        cove_result = run_cove(draft, sources)
        
        # Should find the claim
        assert len(cove_result.claims_extracted) >= 1
        # Should verify it
        verified = [r for r in cove_result.verification_results if r.match]
        assert len(verified) >= 1
    
    def test_flag_unverified_claims(self):
        """Should flag claims not found in sources."""
        draft = """The deal was worth $999 billion."""
        
        sources = [
            {"title": "Major deal announced", "summary": "The agreement totals $50 billion."}
        ]
        
        cove_result = run_cove(draft, sources)
        
        # Should flag as potential hallucination
        assert len(cove_result.hallucination_flags) >= 1
        assert cove_result.has_critical_hallucinations
    
    def test_empty_draft_no_claims(self):
        """Empty or simple draft should have no claims."""
        draft = """Things are happening. The situation evolves."""
        
        cove_result = run_cove(draft, [])
        
        assert len(cove_result.claims_extracted) == 0
        assert cove_result.hallucination_rate == 0.0


class TestFeedbackRouting:
    """Tests for dual-track feedback routing."""
    
    def test_content_issues_generate_feedback(self):
        """Content issues should generate feedback for Writer."""
        draft = """The situation is developing. Various factors are at play."""
        
        analyst = create_mock_analyst_output("Russia")
        sacred = SacredElements(proper_nouns=[], statistics=[], dates=[], quotes=[], event_ids=[])
        events = []
        
        result = run_content_critic(draft, analyst, sacred, events, region="Russia")
        
        assert result.passed is False
        assert len(result.issues) > 0
        assert len(result.feedback) > 0
    
    def test_style_issues_generate_feedback(self):
        """Style issues should generate feedback for Stylist."""
        writer_draft = """Moscow is constrained. HIGHLY LIKELY to change."""
        styled_draft = """It remains to be seen what will happen. At the end of the day,
        time will tell. The situation was analyzed by experts. The report was released."""
        
        sacred = SacredElements(proper_nouns=[], statistics=[], dates=[], quotes=[], event_ids=[])
        
        result = run_style_critic(styled_draft, writer_draft, sacred, region="Russia")
        
        assert result.passed is False
        assert len(result.issues) > 0
        assert len(result.feedback) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
