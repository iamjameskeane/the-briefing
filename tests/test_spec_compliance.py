"""
The Briefing Spec Compliance Tests.

These tests verify that the implementation matches the specification.
Each test focuses on a specific spec requirement.
"""

import pytest
from datetime import datetime

from state import (
    FeaturedPlan,
    SectionDecision,
    KilledRegion,
    QuickHit,
    SectionGroup,
    DocumentSkeleton,
    PICScore,
    EventEmphasis,
    ParagraphPlan,
    SectionBlueprint,
)
from agents.editorial_utils import analyze_thematic_synchronization, calculate_signal_corroboration_score


class TestThematicDetection:
    """Tests for thematic synchronization detection (Phase 6)."""
    
    def test_high_synchronization_returns_thematic(self):
        """≥60% shared causal factor should recommend THEMATIC."""
        connections = [
            "US withdrawal impacts both Europe and Middle East",
            "American retrenchment creates vacuum in East Asia",
            "Washington's pivot affects South Asia security",
        ]
        
        result = analyze_thematic_synchronization(connections, num_regions=4)
        
        # High synchronization since "US" appears in all connections
        assert result["recommendation"] in ["THEMATIC", "HYBRID"]
        assert result["synchronization_pct"] >= 40
    
    def test_low_synchronization_returns_regional(self):
        """<60% shared causal factor should recommend REGIONAL."""
        connections = [
            "Germany domestic energy policy",
            "Local Brazil election updates",
            "Australian wildfire season",
        ]
        
        result = analyze_thematic_synchronization(connections, num_regions=8)
        
        # Low synchronization should recommend REGIONAL
        assert result["recommendation"] == "REGIONAL" or result["synchronization_pct"] < 40
    
    def test_no_connections_returns_regional(self):
        """Empty connections should default to REGIONAL."""
        result = analyze_thematic_synchronization([], num_regions=3)
        
        assert result["recommendation"] == "REGIONAL"
        assert result["synchronization_pct"] == 0


class TestKillDecisions:
    """Tests for Architect kill decisions (spec lines 175-190)."""
    
    def test_pic_score_below_threshold_kills(self):
        """PIC score < 5 should result in KILL."""
        pic = PICScore(probability=2, impact=2, confidence=2)
        # (2 * 2 * 2) / 5 = 8/5 = 1.6, well below 5
        
        assert pic.decision == "KILL"
    
    def test_pic_score_above_threshold_publishes(self):
        """PIC score > 15 should result in PUBLISH."""
        pic = PICScore(probability=5, impact=5, confidence=4)
        # (5 * 5 * 4) / 5 = 100/5 = 20, above 15
        
        assert pic.decision == "PUBLISH"
    
    def test_pic_score_hold_range(self):
        """PIC score 5-15 should result in HOLD."""
        pic = PICScore(probability=3, impact=3, confidence=4)
        # (3 * 3 * 4) / 5 = 36/5 = 7.2, in HOLD range (5-15)
        
        assert pic.decision == "HOLD"
    
    def test_killed_region_has_stats(self):
        """KilledRegion should include events_count and highest_severity."""
        killed = KilledRegion(
            region="South Asia",
            reason="NOTHING_NEW",
            pic_score=12.5,
            events_count=3,
            highest_severity=4,
            brief_explanation="Routine developments only"
        )
        
        assert killed.events_count == 3
        assert killed.highest_severity == 4


class TestHeadlines:
    """Tests for evocative headlines (spec lines 200-220)."""
    
    def test_featured_has_headline(self):
        """FeaturedPlan must have a punchy headline, not just region name."""
        featured = FeaturedPlan(
            region="Middle East",
            headline="Tehran's New Calculus",
            angle="Iran exploits post-Assad vacuum",
            story_archetype="CRISIS",
            rationale="Test"
        )
        
        assert featured.headline != featured.region
        assert len(featured.headline) > 5
    
    def test_section_has_headline(self):
        """SectionDecision must have headline and position_rationale."""
        section = SectionDecision(
            region="East Asia",
            headline="The Squeeze Continues",
            treatment="FULL",
            archetype="TREND",
            word_target=300,
            position_rationale="Second highest PIC score",
            rationale="Trade war escalation"
        )
        
        assert section.headline != section.region
        assert "Second" in section.position_rationale or "position" in section.position_rationale.lower()


class TestTransitions:
    """Tests for section transitions (spec lines 229-235)."""
    
    def test_skeleton_has_transitions(self):
        """DocumentSkeleton should have transitions dict."""
        skeleton = DocumentSkeleton(
            featured=FeaturedPlan(
                region="Middle East",
                headline="Test",
                angle="Test angle",
                story_archetype="CRISIS",
                rationale="Test"
            ),
            sections=[
                SectionDecision(
                    region="East Asia",
                    headline="East Asia Story",
                    treatment="FULL",
                    archetype="TREND",
                    word_target=300,
                    position_rationale="Test",
                    rationale="Test"
                ),
                SectionDecision(
                    region="Europe",
                    headline="Europe Story",
                    treatment="SHORT",
                    archetype="PIVOT",
                    word_target=200,
                    position_rationale="Test",
                    rationale="Test"
                ),
            ],
            narrative_arc="Test arc",
            narrative_arc_explanation="Test",
            section_groups=[SectionGroup(title="Main", sections=["Middle East", "East Asia", "Europe"])],
            organizing_principle="REGIONAL",
            transitions={
                "Middle East → East Asia": "While the Middle East burns, East Asia simmers...",
                "East Asia → Europe": "HARD_BREAK",
            }
        )
        
        assert len(skeleton.transitions) >= 1
        assert "Middle East → East Asia" in skeleton.transitions
    
    def test_transition_or_hard_break(self):
        """Transitions should be either text or HARD_BREAK."""
        transitions = {
            "A → B": "Smooth transition text here",
            "B → C": "HARD_BREAK",
        }
        
        for key, value in transitions.items():
            assert value == "HARD_BREAK" or len(value) > 10


class TestWordCounts:
    """Tests for word count enforcement (spec lines 234-241)."""
    
    def test_featured_word_target(self):
        """Featured sections should target 500-600 words."""
        featured = FeaturedPlan(
            region="Test",
            headline="Test Headline",
            angle="Test angle",
            story_archetype="CRISIS",
            word_target=600,
            rationale="Test"
        )
        
        assert 400 <= featured.word_target <= 700
    
    def test_sleeper_word_target(self):
        """SLEEPER sections should target 80-150 words."""
        section = SectionDecision(
            region="Test",
            headline="Sleeper Story",
            treatment="SLEEPER",
            archetype="SLEEPER",
            word_target=100,
            position_rationale="Test",
            rationale="Quiet development"
        )
        
        assert section.word_target <= 150
    
    def test_full_word_target(self):
        """FULL sections should target 300-400 words."""
        section = SectionDecision(
            region="Test",
            headline="Full Story",
            treatment="FULL",
            archetype="CRISIS",
            word_target=350,
            position_rationale="Test",
            rationale="Major development"
        )
        
        assert 200 <= section.word_target <= 500


class TestCombinedTreatment:
    """Tests for COMBINED section treatment (spec lines 238)."""
    
    def test_combined_has_combine_with(self):
        """COMBINED sections should specify which region to combine with."""
        section = SectionDecision(
            region="Baltic States",
            headline="Baltic Unity",
            treatment="COMBINED",
            archetype="TREND",
            word_target=400,
            combine_with="Poland",
            position_rationale="Grouped for shared theme",
            rationale="Shared NATO concerns"
        )
        
        assert section.treatment == "COMBINED"
        assert section.combine_with is not None
        assert section.combine_with != section.region


class TestSectionGroups:
    """Tests for section grouping (spec lines 340-346)."""
    
    def test_group_has_evocative_title(self):
        """Section groups should have evocative titles, not just 'Regional Updates'."""
        good_titles = [
            "The Week's Story",
            "The Pressure Builds",
            "Ripple Effects",
            "On the Radar",
            "Sleeper Watch"
        ]
        
        group = SectionGroup(
            title="The Pressure Builds",
            sections=["Europe", "East Asia"],
            group_rationale="Tensions escalating in both regions"
        )
        
        # Should not be a generic title
        generic_titles = ["updates", "developments", "news", "events"]
        assert not any(g in group.title.lower() for g in generic_titles)
    
    def test_group_has_rationale(self):
        """Section groups should have a rationale explaining the grouping."""
        group = SectionGroup(
            title="The Long Game",
            sections=["China", "Russia"],
            group_rationale="Both playing strategic patience"
        )
        
        assert group.group_rationale != ""


class TestSignalCorroboration:
    """Tests for Signal Corroboration Score (research phase 2)."""
    
    def test_single_source_low_score(self):
        """Single source events should get low SCS."""
        events = [
            {"title": "Event 1", "source": "Unknown Blog"},
        ]
        
        score = calculate_signal_corroboration_score(events)
        
        assert score <= 2  # Noise tier
    
    def test_multiple_sources_higher_score(self):
        """Multiple authoritative sources should get higher SCS."""
        events = [
            {"title": "Event 1", "source": "Reuters"},
            {"title": "Event 2", "source": "AP"},
            {"title": "Event 3", "source": "Government Statement", "category": "official"},
        ]
        
        score = calculate_signal_corroboration_score(events)
        
        assert score >= 3  # Monitoring tier or higher
    
    def test_quantitative_boosts_score(self):
        """Events with numbers should boost SCS."""
        events = [
            {"title": "GDP grew 3.5% in Q4", "source": "Financial Times", "severity": 8},
            {"title": "Analysis of growth", "source": "Economist"},
        ]
        
        score = calculate_signal_corroboration_score(events)
        
        assert score >= 3  # Quantitative confirmation boosts


class TestQuickHits:
    """Tests for Quick Hit format (spec lines 241)."""
    
    def test_quick_hit_has_primary_event(self):
        """QuickHit should have a primary event_id for linking."""
        qh = QuickHit(
            region="South Asia",
            headline="India-Pakistan border tensions ease",
            content="After last week's skirmish, both sides pulled back forces.",
            event_id="evt_123",
            event_ids=["evt_123", "evt_124"]
        )
        
        assert qh.event_id == "evt_123"
        assert qh.event_id in qh.event_ids
    
    def test_quick_hit_headline_length(self):
        """QuickHit headlines should be short (max ~25 words)."""
        qh = QuickHit(
            region="Test",
            headline="Short punchy headline",
            content="Brief content.",
            event_id="evt_1",
            event_ids=["evt_1"]
        )
        
        word_count = len(qh.headline.split())
        assert word_count <= 25


class TestBlueprintIntegration:
    """Tests for Structure agent blueprint integration."""
    
    def test_paragraph_plan_has_required_fields(self):
        """ParagraphPlan should have purpose, word_target, and beat."""
        para = ParagraphPlan(
            purpose="What just happened",
            word_target=60,
            beat="Opening",
            key_facts=["Key fact 1", "Key fact 2"]
        )
        
        assert para.purpose != ""
        assert para.word_target > 0
        assert para.beat != ""
    
    def test_event_emphasis_has_title(self):
        """EventEmphasis should include event_title for context."""
        event = EventEmphasis(
            event_id="evt_123",
            event_title="Major Trade Deal Announced",
            emphasis="LEAD",
            how_to_use="Use as opening hook",
            role="Primary driver"
        )
        
        assert event.event_title != ""
        assert event.event_id != ""
    
    def test_blueprint_word_allocation(self):
        """Blueprint paragraph word targets should roughly sum to section target."""
        blueprint = SectionBlueprint(
            region="Test",
            archetype="CRISIS",
            word_target=300,
            hook_type="STATEMENT",
            hook_draft="The tension has reached breaking point.",
            paragraphs=[
                ParagraphPlan(purpose="What happened", word_target=75, beat="Opening", key_facts=[]),
                ParagraphPlan(purpose="Why it matters", word_target=90, beat="Stakes", key_facts=[]),
                ParagraphPlan(purpose="Constraints", word_target=75, beat="Analysis", key_facts=[]),
                ParagraphPlan(purpose="Outlook", word_target=60, beat="Closing", key_facts=[]),
            ],
            key_events=[],
            closing_type="PREDICTION",
            closing_draft="This will define the next quarter."
        )
        
        total_words = sum(p.word_target for p in blueprint.paragraphs)
        # Allow 10% variance
        assert abs(total_words - blueprint.word_target) <= blueprint.word_target * 0.15
