# ClassPass Predictor 🎓📊

# A three-class student success analytics pipeline built with modern Python & ML Libraries

📡 Designed to flag Dropout, Enrolled, and Graduate outcomes early using custom ML models and
transparent explainability workflows.

# Guiding Student Success at Scale 🌐

## Project Description 🗺️

ClassPass delivers an interpretable enrollment-time risk detector for higher education programs.
The system pairs handcrafted k-Nearest Neighbors and Decision Tree classifiers with calibration
and cost-sensitive evaluation, highlighting both neighborhood exemplars and global rule-based
insights. Our pipeline ingests the UCI _Predict Students' Dropout and Academic Success_ dataset,
audits class imbalance and missingness, and produces reproducible figures for macro-F1, precision–
recall behavior, and reliability.

## Demo:

https://github.com/arieltyson/class-pass

## Quickstart 🧭

Spin up the baseline experiment from the repository root.

```bash
python -m venv .venv                             # Create isolated environment
source .venv/bin/activate                        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                  # Install project dependencies
python -m scripts.train_baseline \
  --data data/raw/students.csv \
  --target Target \
  --k 7 \
  --scaler standard \
  --distance euclidean                           # Train baseline + write figures/artifacts
```

### Data Audit & Preprocessing Check

Generate EDA summaries and sanity-check the preprocessing pipeline before training:

```bash
python -m scripts.run_eda \
  --data data/raw/students.csv \
  --target Target \
  --outdir reports/eda
```

- Outputs JSON/CSV artifacts describing class balance, missingness, numeric stats, and categorical top values.
- Verifies that the preprocessing stack (encoding, scaling, stratified splits) runs without errors.

### Expected Output

Running the baseline script prints experiment metrics and saves figures under
`reports/figures/`.

- Console shows macro-F1, Brier score, per-class Average Precision, and a confusion matrix.
- `reports/figures/cm_baseline.png` visualizes class-level confusion counts.
- `reports/figures/pr_curves_baseline.png` provides per-class precision–recall curves.
- `reports/figures/reliability_baseline.png` plots probability calibration across bins.
- `artifacts.json` captures run metadata (k, distance, scaler) plus summary metrics.

### Prerequisites 📋

- Python 3.11+
- pip 23+
- Virtualenv support (recommended)
- Access to the UCI dataset CSV placed at `data/raw/students.csv`

### Frameworks

- Python standard library (`argparse`, `pathlib`, `json`)
- NumPy
- pandas
- scikit-learn
- Matplotlib
- tqdm (progress indicators)

### Packages

- joblib (artifact persistence and caching)
- pytest (unit testing)
- black (source formatting)
- ruff (linting)

## Skills Demonstrated 🖌️

ClassPass emphasizes interpretable machine learning for academic risk detection, showcasing how
classical models and evaluation techniques can be orchestrated for reliable early-warning systems.

- **Data Pipeline**: Reproducible ingestion, stratified train/val/test splitting, and dual-mode
  scaling with one-hot encoding for mixed feature types.
- **Custom Models**: From-scratch kNN with configurable distance metrics and explainable neighbor
  lookups; forthcoming Decision Tree with entropy/Gini splits, pruning, and max-depth controls.
- **Explainability**: Local exemplar retrieval from kNN, plus global rule summaries and feature
  importance planned for the tree.
- **Evaluation Rigor**: Macro-F1, per-class PR curves, reliability diagrams, Brier scores, and
  cost-sensitive threshold audits guided by class imbalance.
- **Automation**: Continuous integration (black, ruff, pytest) and scripted training hooks for
  future nested cross-validation runs.

## Contributing ⚙️

We welcome insightful pull requests, benchmarking ideas, and reporting improvements. Fork the
repository, branch from `dev`, and submit focused, well-tested changes. Please include figures or
metric tables when altering modeling behavior so the team can validate performance shifts quickly.

## License 🪪

This project is distributed under the [MIT License](LICENSE) © 2025 ClassPass team (Ariel Tyson &
Phil Akagu-Jones). Feel free to use, modify, and share the code with attribution.

---
