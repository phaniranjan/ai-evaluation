# Code Standards & Quality Guidelines

This project enforces consistent code quality and style standards through centralized configuration.

## Setup

### Installation

```bash
# Install dependencies
make install

# Install development tools
make install-dev

# Setup pre-commit hooks (auto-run checks before commit)
make pre-commit-install
```

## Code Quality Tools

### 1. **Black** - Code Formatter
Automatically formats Python code to a consistent style.
- Line length: 100 characters
- Configuration: `pyproject.toml` (auto-loaded)

```bash
make format
```

### 2. **isort** - Import Sorting
Organizes and sorts imports consistently.
- Configuration: `pyproject.toml` (auto-loaded)
- Runs automatically with formatter

### 3. **Pylint** - Code Analysis
Deeper code quality analysis for maintainability.
- Configuration: `pyproject.toml` (auto-loaded)
- Checks for complexity, naming conventions, etc.

### 4. **Mypy** - Type Checking
Static type checker for Python.
- Configuration: `pyproject.toml` (auto-loaded)
- Helps catch type-related errors

### 5. **Pre-commit Hooks**
Automatically runs checks before each commit.
- Configuration: `.pre-commit-config.yaml`
- Prevents committing code that violates standards

## Common Commands

```bash
# Format code (runs black + isort)
make format

# Run all linters
make lint

# Run all tests
make test

# Run only dynamic LLM tests
make test-dynamic

# Run only static data tests
make test-static

# Clean up cache files
make clean

# Show all available commands
make help
```

## Configuration

All tool configurations are centralized in **`pyproject.toml`** for simplicity and maintainability:

| Tool | Config Section |
|------|-----------------|
| Black | `[tool.black]` |
| isort | `[tool.isort]` |
| Pylint | `[tool.pylint.*]` |
| Mypy | `[tool.mypy]` |
| Pytest | `[tool.pytest.ini_options]` |

This eliminates the need for separate config files (`.flake8`, `.pylintrc`, etc.) and keeps the project configuration DRY.

## Style Guidelines

### Line Length
- Maximum: 100 characters
- Applies to Python, JSON, YAML
- Configured in `pyproject.toml` and `.editorconfig`

### Indentation
- **Python**: 4 spaces
- **JSON/YAML**: 2 spaces
- Configured in `.editorconfig` (cross-editor standard)

### Import Organization
Using `isort` with Black-compatible profile:
1. Standard library imports
2. Third-party imports
3. Local application imports

### Test Markers

Tests are marked for selective execution:
- `@pytest.mark.static` - Static data tests (no LLM calls)
- `@pytest.mark.dynamic` - Dynamic LLM-based tests

```bash
# Run only dynamic tests
pytest -m dynamic

# Run only static tests
pytest -m static

# Run all except dynamic
pytest -m "not dynamic"
```

## Git Workflow

1. Make code changes
2. Run `make format` to format code
3. Run `make lint` to check for issues
4. Run `make test` to verify functionality
5. Commit with `git commit` (pre-commit hooks run automatically)

### Pre-commit Hooks

The following checks run automatically before each commit:
- Trailing whitespace removal
- File-ending fixes
- YAML validation
- Large file detection
- Private key detection
- Black formatting
- isort import sorting
- Pylint analysis

To bypass hooks (not recommended):
```bash
git commit --no-verify
```

## Continuous Integration

All checks are enforced in CI/CD pipelines:
- Code formatting
- Linting
- Type checking
- Test execution

## Project Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | **Centralized configuration** for all tools (Black, isort, Pylint, Mypy, Pytest) |
| `.editorconfig` | Editor consistency settings (cross-editor standard) |
| `.pre-commit-config.yaml` | Pre-commit hooks definition |
| `Makefile` | Development commands |
| `requirements-dev.txt` | Development dependencies |

## Troubleshooting

### Pre-commit issues
```bash
# Reinstall hooks
make pre-commit-install

# Run hooks on all files
pre-commit run --all-files
```

### Formatting conflicts
If you have formatter conflicts, run:
```bash
make format
```

This applies Black and isort consistently.

### Type checking errors
```bash
# See detailed type errors
mypy --show-error-codes .
```

### Pylint configuration
All pylint settings are in `pyproject.toml` under `[tool.pylint.*]`. Modify there instead of creating separate files.

