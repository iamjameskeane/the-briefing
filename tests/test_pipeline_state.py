"""Test PipelineState data flow and field usage."""

import sys
from pathlib import Path
import ast

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_pipeline_state_has_event_count_fields():
    """Verify PipelineState has fields to store event counts."""
    from state import PipelineState
    from datetime import datetime, timezone
    
    state = PipelineState(
        run_id="test",
        started_at=datetime.now(timezone.utc),
    )
    
    # Check that these fields exist and have default values
    assert hasattr(state, 'total_event_count'), "PipelineState missing total_event_count field"
    assert hasattr(state, 'total_source_count'), "PipelineState missing total_source_count field"
    assert state.total_event_count == 0, "total_event_count should default to 0"
    assert state.total_source_count == 0, "total_source_count should default to 0"
    
    # Test that we can set values
    state.total_event_count = 485
    state.total_source_count = 3
    assert state.total_event_count == 485
    assert state.total_source_count == 3


def test_assembly_uses_state_event_count():
    """Verify the assembly/methodology section uses state.total_event_count."""
    run_path = Path(__file__).parent.parent / "run.py"
    with open(run_path, 'r') as f:
        source = f.read()
    
    # Check that we're NOT using the old broken pattern
    assert "len(state.source_context)" not in source or \
           "getattr(state, 'total_event_count'" in source, \
           "Assembly should use state.total_event_count, not len(state.source_context)"
    
    # Check that we ARE using the correct pattern
    assert "getattr(state, 'total_event_count'" in source or \
           "state.total_event_count" in source, \
           "Assembly should reference state.total_event_count"


def test_phase1_stores_event_counts():
    """Verify Phase 1 stores event counts in state."""
    run_path = Path(__file__).parent.parent / "run.py"
    with open(run_path, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    # Find the _run_pipeline_inner function
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == '_run_pipeline_inner':
            # Look for assignments to state.total_event_count
            found_event_count_assignment = False
            found_source_count_assignment = False
            
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Attribute):
                            if (isinstance(target.value, ast.Name) and 
                                target.value.id == 'state'):
                                if target.attr == 'total_event_count':
                                    found_event_count_assignment = True
                                elif target.attr == 'total_source_count':
                                    found_source_count_assignment = True
            
            assert found_event_count_assignment, \
                "Phase 1 should store event count with 'state.total_event_count = ...'"
            assert found_source_count_assignment, \
                "Phase 1 should store source count with 'state.total_source_count = ...'"
            return
    
    raise AssertionError("_run_pipeline_inner function not found in run.py")


def test_metadata_uses_stored_counts():
    """Verify metadata generation uses counts from state, not recalculating."""
    run_path = Path(__file__).parent.parent / "run.py"
    with open(run_path, 'r') as f:
        source = f.read()
    
    # The metadata section should use event_count and source_count variables
    # which were stored in state earlier
    assert '"event_count": event_count' in source, \
           "Metadata should use event_count variable"
    assert '"source_count": source_count' in source, \
           "Metadata should use source_count variable"
