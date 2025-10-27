# Linting & Formatting Guide

Use this guide to run Ruff (linter) and Black (formatter) locally so you can catch style issues before pushing code.

## 1. Prerequisites

- Activate the virtual environment: `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\Activate.ps1` (Windows).
- Install dependencies: `pip install -r requirements.txt`.

## 2. Ruff Linting

Run Ruff to detect import ordering, unused code, and other style or correctness issues.

```bash
ruff check .
```

- Use `ruff check path/to/file.py` to lint a single file.
- For most autofixable issues: `ruff check . --fix`.
- To run only specific rule codes: `ruff check . --select I --select F`.

### Common Fixes

- **Import sorting**: `ruff check . --fix --select I`.
- **Unused imports/variables**: address manually by removing dead code.

## 3. Black Formatting

Black enforces consistent formatting.

```bash
black .
```

- To dry-run without modifying files: `black --check .`.
- Format a single file: `black src/classpass/knn.py`.

### Import Ordering Conventions

Follow PEP 8 and Ruff/isort grouping rules when arranging imports:

```python
from __future__ import annotations

# Standard library (alphabetical)
from collections import Counter
from typing import Literal

# Third-party packages
import numpy as np

# Local application imports go here (blank line separating groups)
```

- `from __future__ import ...` always appears immediately after the module docstring and before all other imports.
- Separate standard library, third-party, and local imports with a blank line.
- Within each group, keep imports sorted alphabetically. Ruff will flag deviations (e.g., rule `I001`).

## 4. Suggested Workflow

1. Make changes.
2. Format code: `black .`.
3. Lint code: `ruff check .`.
4. Run tests: `pytest -q`.

Combining linting and formatting in one command:

```bash
ruff check . && black --check .
```

## 5. Editor Integration

Configure your IDE/Editor to run Ruff and Black on save:

- **VS Code**: install the “Ruff” and “Black Formatter” extensions; enable format-on-save.
- **PyCharm**: configure External Tools for Ruff, and use Black via the “BlackConnect” plugin or built-in file watcher.

Keeping lint/format clean locally ensures GitHub Actions passes on the first try and speeds up code reviews.***
