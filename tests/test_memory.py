"""Tests for briefing memory system (previous edition awareness)."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from run import find_previous_edition, _load_published_editions, register_edition
from agents.editor import EditorInput, _format_previous_edition


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_skeleton():
    """Sample DocumentSkeleton for testing."""
    return {
        "featured": {
            "region": "Test Region",
            "source_cluster_id": "cluster_1",
            "headline": "Test Featured Headline",
            "angle": "TRAJECTORY_SHIFT",
            "story_archetype": "LEADER",
            "word_target": 450,
            "rationale": "Test rationale"
        },
        "sections": [
            {
                "region": "EUROPE",
                "source_cluster_id": "cluster_2",
                "headline": "Test Section One",
                "treatment": "FULL",
                "archetype": "CRISIS",
                "word_target": 300,
            },
            {
                "region": "MIDDLE_EAST",
                "source_cluster_id": "cluster_3",
                "headline": "Test Section Two",
                "treatment": "SHORT",
                "archetype": "TREND",
                "word_target": 200,
            }
        ],
        "killed": [
            {
                "region": "cluster_10",
                "reason": "HOLD_FOR_NEXT_WEEK",
                "pic_score": 9.0,
                "brief_explanation": "Deferred for timing"
            },
            {
                "region": "cluster_11",
                "reason": "BELOW_THRESHOLD",
                "pic_score": 4.0,
                "brief_explanation": "Low impact"
            }
        ],
        "quick_hits": [
            {"region": "AMERICAS", "headline": "Quick hit one"},
            {"region": "EAST_ASIA", "headline": "Quick hit two"}
        ],
        "narrative_arc": "Test narrative arc describing the week's theme.",
        "hub_mechanism": "Test Hub Mechanism",
        "hub_explanation": "This is the test hub explanation describing the mechanism.",
        "organizing_principle": "THEMATIC"
    }


@pytest.fixture
def temp_outputs_dir(sample_skeleton):
    """Create temporary outputs directory with test editions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Create editions.json
        editions_data = {
            "schema_version": "1.0",
            "description": "Test editions",
            "editions": [
                {
                    "run_id": "briefing_20260125_110850",
                    "date": "2026-01-25",
                    "status": "published",
                    "notes": "Test edition 1"
                }
            ]
        }
        (outputs_dir / "editions.json").write_text(json.dumps(editions_data, indent=2))
        
        # Create briefing folder with skeleton
        briefing_dir = outputs_dir / "briefing_20260125_110850"
        agent_outputs = briefing_dir / "agent_outputs"
        agent_outputs.mkdir(parents=True)
        
        (agent_outputs / "02b_document_skeleton.json").write_text(
            json.dumps(sample_skeleton, indent=2)
        )
        
        # Create a test/unpublished run (should be ignored)
        test_dir = outputs_dir / "briefing_20260126_120000"
        test_dir.mkdir()
        
        yield outputs_dir


# =============================================================================
# TESTS: _load_published_editions
# =============================================================================


def test_load_published_editions_returns_list(temp_outputs_dir):
    """Should return list of published editions."""
    editions = _load_published_editions(temp_outputs_dir)
    
    assert isinstance(editions, list)
    assert len(editions) == 1
    assert editions[0]["run_id"] == "briefing_20260125_110850"


def test_load_published_editions_filters_by_status(temp_outputs_dir):
    """Should only return editions with status='published'."""
    # Add a draft edition
    editions_file = temp_outputs_dir / "editions.json"
    data = json.loads(editions_file.read_text())
    data["editions"].append({
        "run_id": "briefing_20260126_000000",
        "date": "2026-01-26",
        "status": "draft",
        "notes": "Draft edition"
    })
    editions_file.write_text(json.dumps(data))
    
    editions = _load_published_editions(temp_outputs_dir)
    
    assert len(editions) == 1
    assert all(e["status"] == "published" for e in editions)


def test_load_published_editions_sorts_newest_first(temp_outputs_dir):
    """Should return editions sorted newest first."""
    editions_file = temp_outputs_dir / "editions.json"
    data = json.loads(editions_file.read_text())
    data["editions"].append({
        "run_id": "briefing_20260118_000000",
        "date": "2026-01-18",
        "status": "published",
        "notes": "Older edition"
    })
    editions_file.write_text(json.dumps(data))
    
    editions = _load_published_editions(temp_outputs_dir)
    
    assert len(editions) == 2
    assert editions[0]["run_id"] == "briefing_20260125_110850"  # Newer
    assert editions[1]["run_id"] == "briefing_20260118_000000"  # Older


def test_load_published_editions_empty_when_no_file():
    """Should return empty list when editions.json doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        editions = _load_published_editions(Path(tmpdir))
        assert editions == []


# =============================================================================
# TESTS: find_previous_edition
# =============================================================================


def test_find_previous_edition_returns_skeleton(temp_outputs_dir):
    """Should return skeleton from most recent published edition."""
    result = find_previous_edition(temp_outputs_dir)
    
    assert result is not None
    assert result["run_id"] == "briefing_20260125_110850"
    assert result["date"] == "2026-01-25"
    assert "skeleton" in result
    assert result["skeleton"]["hub_mechanism"] == "Test Hub Mechanism"


def test_find_previous_edition_excludes_current_run(temp_outputs_dir):
    """Should skip the current run if specified."""
    result = find_previous_edition(
        temp_outputs_dir, 
        current_run_id="briefing_20260125_110850"
    )
    
    assert result is None  # Only edition is excluded


def test_find_previous_edition_ignores_unpublished(temp_outputs_dir):
    """Should ignore briefing folders not in editions.json."""
    # The test run briefing_20260126_120000 exists but isn't in editions.json
    result = find_previous_edition(temp_outputs_dir)
    
    assert result["run_id"] == "briefing_20260125_110850"


def test_find_previous_edition_returns_none_when_no_editions():
    """Should return None when no published editions exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_previous_edition(Path(tmpdir))
        assert result is None


# =============================================================================
# TESTS: register_edition
# =============================================================================


def test_register_edition_creates_file():
    """Should create editions.json if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        register_edition(
            outputs_dir,
            run_id="briefing_20260201_000000",
            date="2026-02-01",
            notes="New edition"
        )
        
        editions_file = outputs_dir / "editions.json"
        assert editions_file.exists()
        
        data = json.loads(editions_file.read_text())
        assert len(data["editions"]) == 1
        assert data["editions"][0]["run_id"] == "briefing_20260201_000000"


def test_register_edition_appends_to_existing(temp_outputs_dir):
    """Should append to existing editions.json."""
    register_edition(
        temp_outputs_dir,
        run_id="briefing_20260201_000000",
        date="2026-02-01",
        notes="New edition"
    )
    
    data = json.loads((temp_outputs_dir / "editions.json").read_text())
    assert len(data["editions"]) == 2


def test_register_edition_skips_duplicate(temp_outputs_dir):
    """Should not add duplicate run_id."""
    register_edition(
        temp_outputs_dir,
        run_id="briefing_20260125_110850",  # Already exists
        date="2026-01-25",
    )
    
    data = json.loads((temp_outputs_dir / "editions.json").read_text())
    assert len(data["editions"]) == 1  # Still just one


# =============================================================================
# TESTS: _format_previous_edition
# =============================================================================


def test_format_previous_edition_includes_hub(sample_skeleton):
    """Should include hub mechanism in output."""
    previous = {
        "skeleton": sample_skeleton,
        "date": "2026-01-25",
        "run_id": "briefing_20260125_110850"
    }
    
    output = _format_previous_edition(previous)
    
    assert "Test Hub Mechanism" in output
    assert "hub explanation" in output.lower()


def test_format_previous_edition_includes_stories(sample_skeleton):
    """Should include published stories table."""
    previous = {
        "skeleton": sample_skeleton,
        "date": "2026-01-25",
        "run_id": "briefing_20260125_110850"
    }
    
    output = _format_previous_edition(previous)
    
    assert "Test Featured Headline" in output
    assert "FEATURED" in output
    assert "Test Section One" in output
    assert "FULL" in output


def test_format_previous_edition_includes_held_clusters(sample_skeleton):
    """Should highlight HOLD_FOR_NEXT_WEEK clusters."""
    previous = {
        "skeleton": sample_skeleton,
        "date": "2026-01-25",
        "run_id": "briefing_20260125_110850"
    }
    
    output = _format_previous_edition(previous)
    
    assert "HELD FOR THIS WEEK" in output
    assert "cluster_10" in output
    assert "Deferred for timing" in output


def test_format_previous_edition_includes_kill_count(sample_skeleton):
    """Should show count of killed clusters."""
    previous = {
        "skeleton": sample_skeleton,
        "date": "2026-01-25",
        "run_id": "briefing_20260125_110850"
    }
    
    output = _format_previous_edition(previous)
    
    assert "Killed last week" in output
    assert "1 cluster" in output  # One killed (the other is HOLD)


def test_format_previous_edition_includes_date(sample_skeleton):
    """Should include date in header."""
    previous = {
        "skeleton": sample_skeleton,
        "date": "2026-01-25",
        "run_id": "briefing_20260125_110850"
    }
    
    output = _format_previous_edition(previous)
    
    assert "2026-01-25" in output


# =============================================================================
# TESTS: EditorInput
# =============================================================================


def test_editor_input_accepts_previous_edition(sample_skeleton):
    """EditorInput should accept previous_edition parameter."""
    previous = {
        "skeleton": sample_skeleton,
        "date": "2026-01-25",
        "run_id": "briefing_20260125_110850"
    }
    
    editor_input = EditorInput(
        thematic_clusters=[],
        hub_candidates=[],
        total_word_budget=2000,
        previous_edition=previous
    )
    
    assert editor_input.previous_edition is not None
    assert editor_input.previous_edition["skeleton"]["hub_mechanism"] == "Test Hub Mechanism"


def test_editor_input_previous_edition_optional():
    """previous_edition should be optional (None by default)."""
    editor_input = EditorInput(
        thematic_clusters=[],
        hub_candidates=[],
        total_word_budget=2000,
    )
    
    assert editor_input.previous_edition is None


# =============================================================================
# INTEGRATION TEST
# =============================================================================


def test_full_flow_find_and_format(temp_outputs_dir):
    """Test finding previous edition and formatting it."""
    # Find
    previous = find_previous_edition(temp_outputs_dir)
    assert previous is not None
    
    # Format
    output = _format_previous_edition(previous)
    assert "PREVIOUS EDITION" in output
    assert "Test Hub Mechanism" in output
    assert "Test Featured Headline" in output
    
    # Use in EditorInput
    editor_input = EditorInput(
        thematic_clusters=[{"cluster_id": 1, "events": []}],
        hub_candidates=[],
        total_word_budget=2000,
        previous_edition=previous
    )
    assert editor_input.previous_edition["skeleton"]["narrative_arc"] == "Test narrative arc describing the week's theme."
