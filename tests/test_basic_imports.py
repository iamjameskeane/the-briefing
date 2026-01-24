"""Basic smoke tests to ensure modules can be imported without errors."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import_run_module():
    """Test that run.py can be imported without NameErrors."""
    try:
        import run
        assert run is not None
    except NameError as e:
        raise AssertionError(f"NameError in run.py: {e}")


def test_import_config():
    """Test that config module can be imported."""
    try:
        import config
        assert config is not None
        assert hasattr(config, 'get_config')
        assert hasattr(config, 'reset_config')
    except Exception as e:
        raise AssertionError(f"Error importing config: {e}")


def test_import_agents():
    """Test that agents module can be imported."""
    try:
        import agents
        assert agents is not None
    except Exception as e:
        raise AssertionError(f"Error importing agents: {e}")


def test_import_cluster():
    """Test that cluster module can be imported."""
    try:
        import cluster
        assert cluster is not None
    except Exception as e:
        raise AssertionError(f"Error importing cluster: {e}")


def test_import_state():
    """Test that state module can be imported."""
    try:
        import state
        assert state is not None
    except Exception as e:
        raise AssertionError(f"Error importing state: {e}")
