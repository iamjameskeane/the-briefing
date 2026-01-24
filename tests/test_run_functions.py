"""Test run.py function signatures and basic validation."""

import sys
from pathlib import Path
import inspect
import ast

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_run_architect_has_config():
    """Verify run_architect function properly gets config."""
    run_path = Path(__file__).parent.parent / "run.py"
    with open(run_path, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    # Find run_architect function
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == 'run_architect':
            # Check that function body contains config = get_config()
            has_get_config = False
            uses_config = False
            
            for child in ast.walk(node):
                # Check for config = get_config()
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id == 'config':
                            if isinstance(child.value, ast.Call):
                                if isinstance(child.value.func, ast.Name):
                                    if child.value.func.id == 'get_config':
                                        has_get_config = True
                
                # Check for config.something usage
                if isinstance(child, ast.Attribute):
                    if isinstance(child.value, ast.Name) and child.value.id == 'config':
                        uses_config = True
            
            if uses_config:
                assert has_get_config, \
                    "run_architect uses 'config' but doesn't call get_config()"
            return
    
    raise AssertionError("run_architect function not found")


def test_all_async_functions_with_config_call_get_config():
    """Verify all async functions that use config call get_config()."""
    run_path = Path(__file__).parent.parent / "run.py"
    with open(run_path, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    functions_using_config_incorrectly = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            func_name = node.name
            has_get_config = False
            uses_config = False
            
            for child in ast.walk(node):
                # Check for config = get_config()
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id == 'config':
                            if isinstance(child.value, ast.Call):
                                if isinstance(child.value.func, ast.Name):
                                    if child.value.func.id == 'get_config':
                                        has_get_config = True
                
                # Check for config.something usage
                if isinstance(child, ast.Attribute):
                    if isinstance(child.value, ast.Name) and child.value.id == 'config':
                        uses_config = True
            
            if uses_config and not has_get_config:
                functions_using_config_incorrectly.append(func_name)
    
    if functions_using_config_incorrectly:
        raise AssertionError(
            f"Functions using 'config' without calling get_config(): "
            f"{', '.join(functions_using_config_incorrectly)}"
        )


def test_run_architect_signature():
    """Test that run_architect has the expected signature."""
    import run
    
    sig = inspect.signature(run.run_architect)
    params = list(sig.parameters.keys())
    
    assert 'state' in params, "run_architect missing 'state' parameter"
    assert 'orchestrator' in params, "run_architect missing 'orchestrator' parameter"
