# ClassPass Project Changelog

## 2025-10-15 — Initial Proposal (Rejected)
- **Scope**: Binary classifier predicting `Continue` vs. `At Risk` using a custom kNN built from scratch; milestones centered on encoding/scaling, validation-based k tuning, and accuracy/F1/confusion matrix reporting.
- **Explainability Plan**: Intended to surface top-k neighbors (labels, distances) to justify predictions.
- **Feedback**: Proposal deemed too narrow—only one model, limited evaluation (no calibration or class-imbalance strategy), and insufficient roadmap for reproducibility or comparative analysis.
- **Outcome**: Team agreed to broaden project goals before the next submission.

## 2025-10-26 — Revised Proposal (Approved)
- **Scope Expansion**: Added a custom Decision Tree alongside kNN, committing to neighbor exemplars and rule-path explanations.
- **Rigor Upgrades**: Introduced plans for nested 5×5 cross-validation, per-class PR-AUC, Brier score, and reliability diagrams to address calibration concerns.
- **Imbalance Strategy**: Documented cost-sensitive thresholds and a resampling audit to better support the minority Dropout class.
- **Tooling Roadmap**: Set expectations for CI, reproducible scripts, and figure exports to back the final demo narrative.

## 2025-11-02 — Milestone 1 Progress Snapshot
- **Data Pipeline**: UCI dataset ingested; categorical encoding and numeric scaling verified; stratified train/val/test split in place.
- **EDA Automation**: Added `scripts/run_eda.py` to produce missingness, class-balance, and preprocessing sanity reports (saved under `reports/eda/`).
- **Baseline Model**: Custom kNN (Euclidean/Manhattan) operational with deterministic tie-breaking and neighbor_explanations for local interpretability.
- **Evaluation Assets**: Macro-F1, confusion matrix, per-class precision–recall curves, Brier score, and reliability plots generated via `scripts/train_baseline.py`.
- **Repo Engineering**: README, requirements, CI workflow (ruff, black, pytest), and unit test coverage for kNN established; reports/figures artifacts captured.
- **Next Focus**: Decision Tree implementation, k-d tree acceleration for kNN, and nested cross-validation scaffolding targeted for Milestone 2.
