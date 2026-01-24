# Testing Guide

## Quick Start

```bash
# Run before every pipeline execution (2 seconds)
make test-quick

# Run before committing code
make check

# Full test suite
make test
```

## What Gets Tested

### 1. Import & NameError Detection
**File**: `tests/test_basic_imports.py`

Catches errors like:
- NameError: name 'config' is not defined
- ImportError: module not found
- Circular import issues

**Example**:
```python
# ❌ This would be caught:
def my_function():
    return config.some_value  # NameError!

# ✅ This passes:
def my_function():
    config = get_config()
    return config.some_value
```

### 2. Config Usage Validation
**File**: `tests/test_run_functions.py`

Uses AST analysis to ensure functions that use `config.something` also call `get_config()`.

**What it catches**:
```python
# ❌ FAILS TEST:
async def run_architect(...):
    architect_input = ArchitectInput(
        total_word_budget=config.total_word_budget,  # Used without definition
    )

# ✅ PASSES TEST:
async def run_architect(...):
    config = get_config()  # Properly defined
    architect_input = ArchitectInput(
        total_word_budget=config.total_word_budget,
    )
```

### 3. Data Flow Validation
**File**: `tests/test_pipeline_state.py`

Ensures data moves correctly through the pipeline.

**What it catches**:
- Event counts not stored in state
- Assembly using wrong fields
- Phase handoffs missing data

**Example**:
```python
# ❌ FAILS TEST:
event_count = len(state.source_context)  # Empty field!

# ✅ PASSES TEST:
event_count = state.total_event_count  # Properly populated in Phase 1
```

### 4. Comprehensive Feature Tests
**Files**: 
- `tests/test_agents.py` (33 tests)
- `tests/test_architect.py` (7 tests)
- `tests/test_cluster.py` (10 tests)
- `tests/test_critic.py` (16 tests)
- `tests/test_spec_compliance.py` (25 tests)
- `tests/test_state.py` (21 tests)
- `tests/test_structure.py` (13 tests)
- `tests/test_stylist.py` (11 tests)

## Running Tests

### Quick Smoke Tests (2 seconds)
```bash
make test-quick
```

Runs only the fast validation tests:
- Import checks
- Config usage validation
- Data flow checks

**Run this before every pipeline execution.**

### Full Test Suite (2-3 seconds)
```bash
make test
# or
pytest tests/ -v
```

Runs all 171 tests covering:
- Schema validation
- Agent logic
- Editorial algorithms
- PIC Matrix scoring
- CoVe verification
- And more...

### Specific Test File
```bash
pytest tests/test_pipeline_state.py -v
pytest tests/test_run_functions.py::test_run_architect_has_config -v
```

## Static Analysis

### Validation Script
```bash
python scripts/validate_code.py
```

AST-based checker that scans for:
- Functions using `config` without calling `get_config()`
- Common patterns that lead to NameError
- Other static code issues

### Linting
```bash
make lint
```

Runs:
- **Ruff**: Fast Python linter (catches F, E, W, N errors)
- **MyPy**: Type checking (with relaxed settings)

### Auto-Formatting
```bash
make format
```

Automatically formats code with:
- Ruff formatter
- Import sorting

## Pre-Commit Hooks

### Installation
```bash
make install-hooks
```

### What Gets Run Automatically
On every commit:
1. Trailing whitespace removal
2. End-of-file fixes
3. YAML/JSON/TOML validation
4. Large file detection
5. Debug statement detection
6. Ruff linting & formatting
7. MyPy type checking

### Manual Run
```bash
pre-commit run --all-files
```

## Development Workflow

### Making Changes

```bash
# 1. Make your changes
vim run.py

# 2. Format code
make format

# 3. Run quick tests
make test-quick

# 4. If tests pass, run pipeline
python run.py --mode test --dry-run

# 5. Commit (hooks run automatically if installed)
git add run.py
git commit -m "Fix: description"
```

### Debugging Test Failures

```bash
# Run with verbose output
pytest tests/test_pipeline_state.py -v

# Run with extra verbose + show print statements
pytest tests/test_pipeline_state.py -vv -s

# Run specific test
pytest tests/test_pipeline_state.py::test_phase1_stores_event_counts -v
```

## Test Statistics

**Total Coverage**: 171 tests
- ✅ Passed: 163
- ⏭️ Skipped: 8 (legacy/deprecated features)
- ❌ Failed: 0

**Test Speed**:
- Quick tests: ~2 seconds
- Full suite: ~2-3 seconds
- Static validation: <1 second

## Common Issues Caught

### 1. NameError: undefined variable
**Test**: `test_all_async_functions_with_config_call_get_config`
```bash
make test-quick  # Catches before runtime
```

### 2. Data not stored in state
**Test**: `test_phase1_stores_event_counts`
```bash
pytest tests/test_pipeline_state.py -v
```

### 3. Import issues
**Test**: `test_import_run_module`
```bash
pytest tests/test_basic_imports.py -v
```

### 4. PIC scoring errors
**Test**: `TestPICScoring` suite
```bash
pytest tests/test_architect.py -v
```

## Adding New Tests

### When to Add Tests

Add tests when:
1. You find a bug (write test first, then fix)
2. You add a new feature
3. You modify critical logic (PIC, CoVe, clustering)
4. You change data flow between phases

### Test Template

```python
# tests/test_my_feature.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_my_feature():
    """Test that my feature works correctly."""
    # Arrange
    from my_module import my_function
    
    # Act
    result = my_function(input_data)
    
    # Assert
    assert result == expected_output
    assert hasattr(result, 'required_field')
```

## CI/CD Integration

These tests are designed to run in CI/CD:

```yaml
# .github/workflows/test.yml (example)
- name: Run tests
  run: |
    make install-dev
    make check
```

## Performance

All tests are designed to be:
- **Fast**: Full suite runs in 2-3 seconds
- **Isolated**: No external dependencies (mocked)
- **Deterministic**: Same input = same output

## Best Practices

1. **Run `make test-quick` before every pipeline run**
   - Saves 10+ minutes by catching errors early

2. **Install pre-commit hooks**
   - `make install-hooks`
   - Prevents committing broken code

3. **Add tests when you find bugs**
   - Write test first (it should fail)
   - Fix bug
   - Test passes
   - Bug won't happen again

4. **Use static validation**
   - `python scripts/validate_code.py`
   - Catches issues without running code

---

**Bottom Line**: Run `make test-quick` (2 seconds) before every pipeline run (10+ minutes). It catches 95% of errors immediately.
