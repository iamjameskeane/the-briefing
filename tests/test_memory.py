
import os
import json
import pytest
from pathlib import Path
from memory import BriefingIndex, get_briefing_content, format_index_for_prompt
from state import DocumentSkeleton, FeaturedPlan, SectionDecision

# Mock skeleton for testing
def create_mock_skeleton():
    return DocumentSkeleton(
        featured=FeaturedPlan(
            region="Test Region",
            source_cluster_id="cluster_test_1",
            headline="Test Headline",
            angle="CRISIS",
            story_archetype="LEADER",
            word_target=500,
            rationale="Test rationale"
        ),
        sections=[
            SectionDecision(
                region="Section 1",
                source_cluster_id="cluster_test_2",
                headline="Section Headline",
                treatment="FULL",
                archetype="TREND",
                word_target=300,
                rationale="Section rationale",
                position_rationale="First"
            )
        ],
        killed=[],
        quick_hits=[],
        narrative_arc="Test arc",
        narrative_arc_explanation="Test explanation",
        section_groups=[],
        transitions={},
        organizing_principle="THEMATIC",
        hub_mechanism="Test Hub",
        hub_explanation="Test Hub Explanation",
        hub_manifestations=["Region 1"],
        hub_angle="TREND"
    )

def test_index_management(tmp_path):
    index_file = tmp_path / "test_index.json"
    index = BriefingIndex(index_file=str(index_file))
    
    # Test initial state
    assert os.path.exists(index_file)
    assert index.load_index() == []
    
    # Test update_index
    skeleton = create_mock_skeleton()
    index.update_index("run_test_1", "2026-02-07", skeleton)
    
    loaded = index.load_index()
    assert len(loaded) == 1
    assert loaded[0]["run_id"] == "run_test_1"
    assert loaded[0]["hub"] == "Test Hub"
    assert loaded[0]["headlines"]["cluster_test_1"] == "Test Headline"
    assert loaded[0]["headlines"]["cluster_test_2"] == "Section Headline"

def test_content_retrieval(tmp_path, monkeypatch):
    # Setup mock output directory
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "run_test_1"
    agent_outputs = run_dir / "agent_outputs"
    agent_outputs.mkdir(parents=True)
    
    skeleton = create_mock_skeleton()
    skeleton_path = agent_outputs / "02b_document_skeleton.json"
    with open(skeleton_path, "w") as f:
        json.dump(skeleton.model_dump(), f)
        
    # Monkeypatch the outputs directory in memory.py
    monkeypatch.setattr("memory.Path", lambda x: tmp_path / x if x == "outputs" else Path(x))
    
    # Test high-level retrieval
    summary = get_briefing_content("run_test_1")
    assert "Hub: Test Hub" in summary
    assert "Narrative Arc: Test arc" in summary
    
    # Test section retrieval
    section = get_briefing_content("run_test_1", "cluster_test_1")
    assert "HEADLINE: Test Headline" in section
    assert "RATIONALE: Test rationale" in section
    
    # Test error case
    error = get_briefing_content("run_test_1", "non_existent")
    assert "Error" in error

def test_prompt_formatting():
    index_data = [
        {
            "run_id": "run_1",
            "date": "2026-01-25",
            "hub": "Hub 1",
            "headlines": {"c1": "Headline 1"}
        }
    ]
    prompt = format_index_for_prompt(index_data)
    assert "[2026-01-25] Run ID: run_1" in prompt
    assert "Hub: Hub 1" in prompt
    assert "Headline 1 (c1)" in prompt
