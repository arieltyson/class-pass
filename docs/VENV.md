# Virtual Environment Guide

Follow this guide to create and manage a Python virtual environment for the ClassPass project. Using a virtual environment ensures dependencies stay isolated from system-wide packages and makes it easier to reproduce results across machines.

## 1. Prerequisites

- Python 3.11 installed (`python3 --version` should report 3.11.x).
- `pip` upgraded (`python3 -m pip install --upgrade pip`).

## 2. Create the Virtual Environment

From the project root (`class-pass/`):

```bash
python3 -m venv .venv
```

This command creates a `.venv` directory containing an isolated interpreter and site-packages folder.

## 3. Activate the Environment

- **macOS / Linux**

  ```bash
  source .venv/bin/activate
  ```

- **Windows (PowerShell)**

  ```powershell
  .venv\Scripts\Activate.ps1
  ```

Once activated, your shell prompt usually shows `(.venv)` to indicate you are inside the environment.

## 4. Install Project Dependencies

```bash
pip install -r requirements.txt
```

This pulls the exact versions specified for NumPy, pandas, scikit-learn, matplotlib, tqdm, joblib, pytest, black, and ruff.

## 5. Verify Setup

Run smoke checks to confirm tooling works:

```bash
ruff --version
black --version
pytest -q
python scripts/train_baseline.py --help
```

If each command succeeds, your environment is ready for development and testing.

## 6. Deactivate When Finished

```bash
deactivate
```

Deactivation returns your shell to the global Python context without removing the `.venv` directory.

## 7. Common Tips

- Re-run `pip install -r requirements.txt` after pulling updates that modify dependencies.
- To reset the environment completely, remove the `.venv/` directory and repeat steps 2–4.
- Keep `.venv/` out of version control: the directory is ignored through `.gitignore` by default.***
