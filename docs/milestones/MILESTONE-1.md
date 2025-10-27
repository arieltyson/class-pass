# Milestone 1 (Oct 19, 2025)

Milestone 1 established the foundation for ClassPass by loading the UCI dataset, validating preprocessing, implementing a custom kNN baseline, and exporting core evaluation artifacts. All work assumes the team is brand-new to machine learning, so difficulty reflects extra learning overhead.

## Task Dependency Matrix

| Task ID | Title                                                     | Depends On | Difficulty (ML Newcomer) | Status           | Test Coverage                | Notes / Prerequisites                                                                 |
| ------- | --------------------------------------------------------- | ---------- | ------------------------ | ---------------- | ---------------------------- | ------------------------------------------------------------------------------------- |
| T1      | Repository & Environment Scaffold                         | _None_     | Easy                     | ✅ Complete      | N/A (infrastructure)         | Initialize git repo, add `.gitignore`, `requirements.txt`, `pyproject.toml`, CI stub. |
| T2      | Acquire & Inspect Dataset                                 | T1         | Medium                   | 🟡 In Progress   | 0% (manual spot-checks)      | Download UCI CSV, confirm `Target` values, review basic stats for missingness.        |
| T3      | Preprocessing Pipeline (encoding + scaling + splits)      | T2         | Hard                     | ✅ Complete      | 0% (automated coverage TBD)  | Implement loaders, stratified train/val/test split, scaling toggles.                  |
| T4      | Custom kNN Implementation                                 | T3         | Hard                     | ✅ Complete      | ~40% (`tests/test_knn.py`)   | Build kNN with Euclidean/Manhattan support, tie-breaking, neighbor explanations.      |
| T5      | Baseline Training Script (`scripts/train_baseline.py`)    | T4         | Medium                   | ✅ Complete      | 0% (smoke via manual run)    | Wire preprocessing + kNN, produce metrics, save artifacts/figures.                    |
| T6      | Evaluation Plots & Metrics (F1, PR, Brier, Reliability)   | T5         | Medium                   | ✅ Complete      | 0% (visual/manual checks)    | Implement confusion matrix, PR curves, reliability plots, Brier calculation.          |
| T7      | Documentation & Reporting (README, CHANGELOG entries)     | T1         | Easy                     | ✅ Complete      | N/A (docs)                   | Capture setup steps, milestone narrative, GitHub About section alignment.             |
| T8      | CI Setup (ruff, black, pytest workflow)                   | T1         | Medium                   | ✅ Complete      | 100% (CI workflow passing)   | Ensure lint/test jobs run on push/PR, acts as regression safety net.                  |
| T9      | Dataset Placement Policy (no raw data in repo history)    | T2         | Easy                     | ✅ Complete      | N/A (process)                | Document data handling conventions in README and `.gitignore`.                        |
| T10     | Milestone Review & Next-Step Planning                     | T6, T7     | Easy                     | ⬜ Pending       | N/A (meeting)                | Summarize learnings, outline Decision Tree and k-d tree plan for Milestone 2.         |

---

## Workflow Notes

1. **Foundational Setup (T1, T2)**  
   Establish the repo and confirm dataset integrity before touching modeling. New ML practitioners should spend extra time validating column types and class balance to avoid downstream debugging surprises.

2. **Data Processing Core (T3)**  
   Preprocessing is the critical dependency for all modeling tasks. Prioritize writing helper functions with docstrings and, where possible, lightweight unit tests to lock in behavior for stratified splits and scaling.

3. **Model Implementation (T4, T5)**  
   After preprocessing is stable, build the kNN classifier and integrate it into a runnable script. Treat script execution logs as provisional validation until more formal tests are written.

4. **Evaluation Artifacts (T6)**  
   Once predictions work, generate figures and metrics required for grading. Save outputs in `reports/figures/` and ensure plots look sensible (e.g., class names align on axes).

5. **Process & Documentation (T7–T10)**  
   Keep README and changelog updated as work completes. CI ensures repeatability, and the closing review meeting should translate directly into Milestone 2 issues (Decision Tree, k-d tree acceleration, nested CV).
