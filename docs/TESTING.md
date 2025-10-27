# Testing Guide

This document explains how to run the automated checks for ClassPass. Tests are designed to run locally in a virtual environment and automatically in CI.

## 1. Prerequisites

- Activate the project virtual environment (`source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` on Windows).
- Install dependencies: `pip install -r requirements.txt`.

## 2. Run Unit Tests

Use pytest to execute the test suite:

```bash
pytest -q
```

- `-q` runs in quiet mode, showing only minimal output.
- Omit `-q` when you need full tracebacks or verbose details.

Run a specific test file or test case:

```bash
pytest tests/test_knn.py::test_knn_simple
```

## 3. Linting and Formatting Checks

The project uses Ruff and Black to enforce code style.

```bash
ruff check .
black --check .
```

- `ruff check .` lints the entire repository.
- `black --check .` verifies formatting without modifying files.

To auto-format:

```bash
black .
```

## 4. Combined Workflow Script (Optional)

Create a helper command to run all checks in sequence:

```bash
ruff check . && black --check . && pytest -q
```

Consider adding this to a shell alias or a make target for convenience.

## 5. Continuous Integration

Every push and pull request to `main` or `dev` triggers the GitHub Actions workflow (`.github/workflows/python.yml`) which:

1. Installs dependencies via `requirements.txt`.
2. Runs `ruff check .`.
3. Runs `black --check .`.
4. Runs `pytest -q`.

Ensure local runs pass before pushing to keep CI green and avoid blocking teammates.
