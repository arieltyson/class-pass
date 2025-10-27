# Milestone 2 (Due Nov 16, 2025)

Milestone 2 focuses on extending ClassPass beyond the baseline by adding a custom Decision Tree, accelerating kNN with a k-d tree, and standing up rigorous evaluation workflows (nested cross-validation, cost-sensitive thresholds, extended reporting). All Milestone 1 tasks are assumed complete.

## Task Dependency Matrix

| Task ID | Title                                                                | Depends On | Difficulty (ML Newcomer) | Status     | Test Coverage            | Notes / Prerequisites                                                                                      |
| ------- | -------------------------------------------------------------------- | ---------- | ------------------------ | ---------- | ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Task ID | Title                                                                | Depends On | Difficulty (ML Newcomer) | Status            | Test Coverage                | Notes / Prerequisites                                                                                      |
| ------- | -------------------------------------------------------------------- | ---------- | ------------------------ | ----------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------- |
| T11     | Decision Tree Core (entropy/Gini splits, recursion)                  | T4, T5     | Hard                     | ⬜ Not Started    | 0% (planned unit tests)      | Implement from-scratch ID3/CART-style tree with stopping criteria (`max_depth`, `min_samples_leaf`).       |
| T12     | Decision Tree Pruning & Rule Export                                  | T11        | Hard                     | ⬜ Not Started    | 0% (planned integration)     | Add pre/post pruning, generate rule-path explanations, serialize feature importances.                      |
| T13     | Decision Tree Training Script (`scripts/train_tree.py`)              | T11, T12   | Medium                   | ⬜ Not Started    | 0% (planned pytest fixture)  | Mirror baseline CLI workflow with CLI flags, figure exports, and `artifacts_tree.json`.                    |
| T14     | kNN k-d Tree Accelerator                                             | T4         | Hard                     | 🟡 In Progress    | 0% (benchmarks pending)      | Build k-d tree data structure, integrate as optional mode in kNN, include speed/accuracy smoke tests.      |
| T15     | Nested 5×5 Cross-Validation Harness                                  | T5, T13    | Very Hard                | ⬜ Not Started    | 0% (planned end-to-end)      | Implement outer evaluation loop with inner hyperparameter tuning for both kNN and Decision Tree models.    |
| T16     | Cost-Sensitive Threshold Sweep (Dropout emphasis)                    | T15        | Medium                   | ⬜ Not Started    | 0% (manual / notebook)       | Evaluate per-class precision/recall trade-offs, log thresholds and metrics for report inclusion.           |
| T17     | Expanded Evaluation Visualizations (tables, calibration comparisons) | T13, T16   | Medium                   | ⬜ Not Started    | 0% (manual verification)     | Produce consolidated macro-F1/Brier tables, PR curves per model, calibration overlays across CV folds.     |
| T18     | Documentation & Milestone Report (PDF narrative + figures)           | T11–T17    | Medium                   | ⬜ Not Started    | N/A (docs)                   | Prepare brief recap, highlight three accomplishments with screenshots/metrics, package as PDF for submit. |
| T19     | CI Enhancements (benchmark hooks, longer tests toggled)              | T14, T15   | Medium                   | ⬜ Not Started    | 0% (GitHub Actions targets)  | Extend workflow to conditionally run heavier tests/benchmarks (e.g., via `pytest -m slow`).                |
| T20     | Demo Narrative Updates (CHANGELOG, Milestone 2 slide prep)           | T18        | Easy                     | ⬜ Not Started    | N/A (docs)                   | Keep docs current, capture learnings for final presentation storyline.                                     |

---

## Workflow Notes

1. **Decision Tree Track (T11–T13)**  
   Start by implementing the tree classifier core, ensuring unit coverage on split calculations and stopping logic. Follow with pruning strategies and rule export, then wrap in a training script parallel to the kNN baseline for consistency.

2. **Performance Enhancements (T14, T19)**  
   Build the k-d tree accelerator early so integration challenges surface while other work is in flight. Add lightweight benchmarks (wall-clock comparisons on synthetic datasets) and wire new checks into CI once stable.

3. **Evaluation Rigor (T15–T17)**  
   Develop the nested cross-validation scaffold after both models have runnable scripts. Use configuration files or CLI flags to manage hyperparameters. Generate comparative tables and calibration plots as artifacts for the report.

4. **Cost-Sensitive Analysis (T16)**  
   After CV metrics exist, sweep Dropout thresholds to illustrate the trade-off story. Capture key plots/tables for the submission package.

5. **Reporting & Narrative (T18–T20)**  
   Assemble the milestone PDF summarizing accomplishments with evidence. Update changelog and presentation materials to reflect new capabilities and lessons learned, setting up the final demo narrative.
