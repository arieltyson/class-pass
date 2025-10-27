# Manual Test: EDA & Preprocessing Verification

Use this checklist to manually validate the exploratory data analysis workflow.

## 1. Set Up Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Ensure Dataset Is Available

Place the CSV at `data/raw/students.csv` with a `Target` column containing `Dropout`, `Enrolled`, and `Graduate`.

Verify the file exists:

```bash
ls data/raw/students.csv
```

## 3. Run EDA Script

Execute the EDA pipeline to generate summary artifacts.

```bash
python -m scripts.run_eda \
  --data data/raw/students.csv \
  --target Target \
  --outdir reports/eda
```

## 4. Inspect Outputs

Confirm the expected files are created:

```bash
ls reports/eda
cat reports/eda/summary.json
head reports/eda/missingness.csv
head reports/eda/numeric_summary.csv
cat reports/eda/categorical_top_values.json
```

- `summary.json` should report dataset size, feature types, target distribution, and preprocessing split shapes.
- `missingness.csv` lists per-column missing counts and percentages (descending order).
- `numeric_summary.csv` contains descriptive statistics for numeric features.
- `categorical_top_values.json` records top categories for each non-numeric feature.

## 5. Spot-Check Preprocessing

Confirm scaling and encoding run without errors by re-running the script with a different scaler:

```bash
python -m scripts.run_eda \
  --data data/raw/students.csv \
  --target Target \
  --outdir reports/eda_minmax \
  --scaler minmax
```

Inspect the new folder to ensure artifacts are generated and column counts remain consistent.

## 6. Cleanup (Optional)

Remove generated artifacts if not needed:

```bash
rm -rf reports/eda reports/eda_minmax
deactivate
```
