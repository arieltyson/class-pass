# 📘 **ClassPass – Early Student Risk Detection (Custom ML Pipeline)**

**ClassPass** is a fully reproducible, interpretable machine-learning pipeline for early dropout-risk detection in higher education programs.  
It implements:

- A **custom from-scratch k-Nearest Neighbors classifier**  
- A **preprocessing and encoding pipeline**  
- **EDA + data auditing tools**  
- Train/validation/test splitting with hyperparameter tuning  
- **Explainability** via neighbor inspection  
- Model evaluation (F1, macro-F1, confusion matrix)

The system uses the UCI *Predict Students’ Dropout and Academic Success* dataset and reduces the original 3-class label into a binary target:

- **At Risk** (Dropout)  
- **Continue** (Enrolled + Graduate)

Everything is implemented from first principles — no sklearn KNN, no automated pipelines — to demonstrate understanding of ML fundamentals and interpretability.

---

# 🚀 **Features**

### **🔧 Custom ML Model**
- Handmade **kNN classifier** supporting:
  - Euclidean / Manhattan distance  
  - Soft probability estimation  
  - Local neighbor explanations:
    > “3 out of your 5 most similar students were At Risk.”

### **📊 EDA & Data Auditing**
- Missingness report  
- Class distribution  
- Top categorical values  
- Numerical summary statistics  
- Outputs JSON + CSV artifacts

### **🧹 Preprocessing Pipeline**
- Automatic feature-type detection (numeric vs categorical)
- One-hot encoding  
- Standard or MinMax scaling  
- Stratified train/validation/test splitting  

### **📈 Evaluation Tools**
- F1 (binary + macro)  
- Accuracy  
- Confusion matrix plot  
- Validation sweep over k  
- F1 vs k plot  
- Artifacts JSON saved for reproducibility  

### **🧪 Full Testing Suite**
- Tests for:
  - EDA structure  
  - kNN correctness  
  - Probability outputs  
  - Neighbor explanations  
  - Comparison vs sklearn KNN (sanity-check)

---

# 📁 **Project Structure**

```
class-pass/
│
├── src/classpass/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── knn.py
│   └── evaluation.py
│
├── scripts/
│   ├── run_eda.py
│   └── train_baseline.py
│
├── data/
│   └── raw/
│       └── students.csv
│
├── reports/
│   ├── eda/
│   └── figures/
│
├── tests/
│   ├── test_eda.py
│   ├── test_knn.py
│   └── __init__.py
│
├── CMPT-310-Proposal.pdf
├── README.md
├── requirements.txt
├── pyproject.toml
└── pytest.ini
```

---

# 🧰 **Installation**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Place the UCI dataset in:

```
data/raw/students.csv
```

---

# 🔎 **Run EDA**

```bash
python -m scripts.run_eda   --data data/raw/students.csv   --target Target   --binary   --outdir reports/eda
```

---

# 🤖 **Train Baseline kNN**

```bash
python -m scripts.train_baseline   --data data/raw/students.csv   --target Target   --binary   --scaler standard   --distance euclidean   --k-grid 3,5,7,9,11   --outdir reports/figures
```

---

# 🧪 **Run Tests**

```bash
pytest -q
```

---

# 📈 Example Performance

```
[Train] Best k: 3
Validation F1(At Risk): ~0.78

[Test metrics]
  accuracy:   0.896
  f1_binary:  0.819
  f1_macro:   0.873
```

---

# 📄 **License**

MIT License © 2025  
Ariel Tyson & Phil Akagu-Jones  
