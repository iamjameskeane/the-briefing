"""
Tests for The Briefing clustering module, including thematic-first clustering.
"""

import numpy as np
import pytest

from cluster import (
    ThematicCluster,
    ClusterAnalysis,
    determine_recommended_organization,
    create_event_text,
    determine_trend,
)


class TestThematicCluster:
    """Tests for ThematicCluster dataclass."""
    
    def test_basic_creation(self):
        """Test creating a thematic cluster."""
        cluster = ThematicCluster(
            cluster_id=0,
            theme_label="US Policy Shifts",
            events=[{"id": "1", "title": "Test"}],
            event_ids=["1"],
            regions_touched=["EUROPE", "MIDDLE_EAST", "EAST_ASIA"],
            region_event_counts={"EUROPE": 2, "MIDDLE_EAST": 3, "EAST_ASIA": 1},
            avg_severity=7.5,
            is_hub_candidate=True,
        )
        
        assert cluster.theme_label == "US Policy Shifts"
        assert len(cluster.regions_touched) == 3
        assert cluster.is_hub_candidate is True
    
    def test_to_dict(self):
        """Test serialization to dict."""
        cluster = ThematicCluster(
            cluster_id=1,
            theme_label="Energy Crisis",
            events=[{"id": "1"}, {"id": "2"}],
            event_ids=["1", "2"],
            regions_touched=["EUROPE", "AMERICAS"],
            region_event_counts={"EUROPE": 1, "AMERICAS": 1},
            avg_severity=6.0,
            is_hub_candidate=False,
        )
        
        d = cluster.to_dict()
        assert d["theme_label"] == "Energy Crisis"
        assert d["event_count"] == 2
        assert d["is_hub_candidate"] is False
        assert d["avg_severity"] == 6.0


class TestClusterAnalysisThematic:
    """Tests for thematic fields in ClusterAnalysis."""
    
    def test_includes_thematic_fields(self):
        """Test that ClusterAnalysis has thematic fields."""
        analysis = ClusterAnalysis(
            regions={},
            thematic_clusters=[],
            hub_candidates=[],
            recommended_organization="REGIONAL",
        )
        
        assert hasattr(analysis, 'thematic_clusters')
        assert hasattr(analysis, 'hub_candidates')
        assert hasattr(analysis, 'recommended_organization')
    
    def test_to_dict_includes_thematic(self):
        """Test that to_dict includes thematic fields."""
        hub = ThematicCluster(
            cluster_id=0,
            theme_label="Test Hub",
            events=[{"id": "1"}],
            event_ids=["1"],
            regions_touched=["A", "B", "C"],
            region_event_counts={"A": 1, "B": 1, "C": 1},
            avg_severity=8.0,
            is_hub_candidate=True,
        )
        
        analysis = ClusterAnalysis(
            regions={},
            thematic_clusters=[hub],
            hub_candidates=[hub],
            recommended_organization="THEMATIC",
        )
        
        d = analysis.to_dict()
        assert "thematic_clusters" in d
        assert "hub_candidates" in d
        assert "recommended_organization" in d
        assert d["recommended_organization"] == "THEMATIC"
        assert len(d["hub_candidates"]) == 1


class TestDetermineRecommendedOrganization:
    """Tests for organization recommendation logic."""
    
    def test_no_clusters_returns_regional(self):
        """Empty thematic clusters should recommend REGIONAL."""
        result = determine_recommended_organization([], 100)
        assert result == "REGIONAL"
    
    def test_no_hub_candidates_returns_regional(self):
        """Clusters without hub candidates should recommend REGIONAL."""
        non_hub = ThematicCluster(
            cluster_id=0,
            theme_label="Test",
            events=[{"id": "1"}],
            event_ids=["1"],
            regions_touched=["A", "B"],  # Only 2 regions
            region_event_counts={"A": 1, "B": 1},
            avg_severity=5.0,
            is_hub_candidate=False,
        )
        
        result = determine_recommended_organization([non_hub], 50)
        assert result == "REGIONAL"
    
    def test_strong_hub_returns_thematic(self):
        """Strong hub (4+ regions, high severity) should recommend THEMATIC."""
        strong_hub = ThematicCluster(
            cluster_id=0,
            theme_label="Major Theme",
            events=[{"id": str(i), "severity": 8} for i in range(10)],
            event_ids=[str(i) for i in range(10)],
            regions_touched=["A", "B", "C", "D"],  # 4 regions
            region_event_counts={"A": 3, "B": 3, "C": 2, "D": 2},
            avg_severity=7.5,  # High severity
            is_hub_candidate=True,
        )
        
        result = determine_recommended_organization([strong_hub], 100)
        assert result == "THEMATIC"
    
    def test_multiple_hubs_returns_hybrid(self):
        """Multiple hub candidates should recommend HYBRID."""
        hub1 = ThematicCluster(
            cluster_id=0,
            theme_label="Hub 1",
            events=[{"id": "1"}],
            event_ids=["1"],
            regions_touched=["A", "B", "C"],
            region_event_counts={"A": 1, "B": 1, "C": 1},
            avg_severity=6.5,
            is_hub_candidate=True,
        )
        hub2 = ThematicCluster(
            cluster_id=1,
            theme_label="Hub 2",
            events=[{"id": "2"}],
            event_ids=["2"],
            regions_touched=["D", "E", "F"],
            region_event_counts={"D": 1, "E": 1, "F": 1},
            avg_severity=6.0,
            is_hub_candidate=True,
        )
        
        result = determine_recommended_organization([hub1, hub2], 100)
        assert result == "HYBRID"
    
    def test_single_weak_hub_returns_hybrid(self):
        """Single hub with 3 regions but lower severity should recommend HYBRID."""
        weak_hub = ThematicCluster(
            cluster_id=0,
            theme_label="Weak Hub",
            events=[{"id": "1"}],
            event_ids=["1"],
            regions_touched=["A", "B", "C"],  # 3 regions
            region_event_counts={"A": 1, "B": 1, "C": 1},
            avg_severity=6.0,  # Not high enough for THEMATIC
            is_hub_candidate=True,
        )
        
        result = determine_recommended_organization([weak_hub], 50)
        assert result == "HYBRID"


class TestDetermineTrend:
    """Tests for trend determination."""
    
    def test_empty_events_stable(self):
        """Empty event list should return STABLE."""
        assert determine_trend([]) == "STABLE"
    
    def test_high_severity_escalating(self):
        """High average severity should return ESCALATING."""
        events = [
            {"severity": 8},
            {"severity": 7},
            {"severity": 8},
        ]
        assert determine_trend(events) == "ESCALATING"
    
    def test_low_severity_de_escalating(self):
        """Low severity should return DE_ESCALATING."""
        events = [
            {"severity": 3},
            {"severity": 4},
            {"severity": 3},
        ]
        assert determine_trend(events) == "DE_ESCALATING"
    
    def test_volatile_high_volume_mixed(self):
        """High volume with mixed severity should return VOLATILE."""
        events = [{"severity": s} for s in [2, 4, 6, 8, 3, 5, 7, 9, 1, 5]]
        assert determine_trend(events) == "VOLATILE"


class TestCreateEventText:
    """Tests for event text creation for embeddings."""
    
    def test_combines_fields(self):
        """Should combine title, summary, location, category."""
        event = {
            "title": "Major Summit",
            "summary": "Leaders meet to discuss",
            "location_name": "Geneva",
            "category": "Diplomacy",
        }
        
        text = create_event_text(event)
        
        assert "Major Summit" in text
        assert "Leaders meet to discuss" in text
        assert "Geneva" in text
        assert "Diplomacy" in text
    
    def test_handles_missing_fields(self):
        """Should handle events with missing fields."""
        event = {"title": "Simple Event"}
        
        text = create_event_text(event)
        
        assert "Simple Event" in text
        assert "Location" not in text  # No location to include
