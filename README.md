# 📘 **ClassPass – Early Student Risk Detection (Custom ML Pipeline)**  

**ClassPass** is a fully reproducible, interpretable machine-learning pipeline for early dropout-risk detection in higher education programs.  
It implements:

- A **custom from-scratch k-Nearest Neighbors classifier**  
- A **custom Decision Tree classifier** (entropy/Gini, interpretable rule extraction)  
- A full preprocessing and encoding pipeline  
- Separate scripts for EDA, kNN training, and Decision Tree training  
- Explainability tools (neighbors + rule extraction)  
- Validation-based model selection (tuning *k* or *max_depth*)  
- Train/val/test splits, metrics, confusion matrices, and artifacts

The system uses the UCI *Predict Students’ Dropout and Academic Success* dataset and maps the original 3-class target into a binary label:

- **At Risk** (Dropout)  
- **Continue** (Enrolled + Graduate)

Everything is implemented from first principles to demonstrate clear ML fundamentals, interpretability, and reproducibility.

---

# 🚀 **Features**

### **🔧 Custom ML Models**
#### **kNN (from scratch)**
- Euclidean / Manhattan distance  
- Predict + predict_proba  
- Local neighbor explanations  

#### **Decision Tree (from scratch)**
- Entropy or Gini impurity  
- Information gain  
- Customizable max depth + min samples split  
- Human-readable rule extraction  
- Predict + predict_proba  

---

### **📊 EDA & Data Auditing**
- Missingness report  
- Class balance  
- Basic numeric statistics  
- Top categorical values  
- JSON + CSV outputs

---

### **🧹 Preprocessing**
- Automatic detection of numerical vs. categorical features  
- One-hot encoding  
- Standard or MinMax scaling  
- Stratified train/val/test splitting  

---

### **📈 Evaluation Tools**
- F1 (binary + macro)  
- Accuracy  
- Confusion Matrix (with saved PNG)  
- F1 vs k plot  
- Artifacts JSON  

---

### **🧪 Testing**
- `test_eda.py` — validates EDA summary structure  
- `test_knn.py` — validates kNN correctness  
- `test_decision_tree.py` — validates tree predictions + rule generation  

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
│   ├── decision_tree.py
│   └── evaluation.py
│
├── scripts/
│   ├── run_eda.py
│   ├── train_baseline.py
│   └── run_tree.py
│
├── data/
│   └── raw/
│       └── students.csv
│
├── reports/
│   ├── eda/
│   ├── figures/
│   └── figures_tree/
│
├── tests/
│   ├── test_eda.py
│   ├── test_knn.py
│   ├── test_decision_tree.py
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

Place the dataset at:

```
data/raw/students.csv
```

The loader automatically handles semicolon-delimited UCI CSVs.

---

# 🔎 **1. Run EDA**

```bash
python -m scripts.run_eda   --data data/raw/students.csv   --target Target   --binary   --outdir reports/eda
```

---

# 🤖 **2. Train Baseline kNN**

```bash
python -m scripts.train_baseline   --data data/raw/students.csv   --target Target   --binary   --scaler standard   --distance euclidean   --k-grid 3,5,7,9,11   --outdir reports/figures
```

Outputs:

- `cm_knn.png`  
- `f1_vs_k.png`  
- `artifacts_knn.json`

---

# 🌳 **3. Train Decision Tree (NEW)**

```bash
python -m scripts.run_tree   --data data/raw/students.csv   --target Target   --binary   --criterion entropy   --depth-grid 3,5,7,9   --min-samples-split 2   --outdir reports/figures_tree
```

Outputs:

- `cm_tree.png`  
- `tree_rules.txt`  
- `tree_artifacts.json`  

---

# 🧪 **4. Run Tests**

```bash
pytest -q
```

Expected:

```
7 passed in X.XXs
```

---

# 📈 Example kNN Performance

```
Validation F1(At Risk): ~0.78
Test accuracy:     ~0.896
Test F1_binary:    ~0.819
Test F1_macro:     ~0.873
```

---

# 📄 **License**

MIT License © 2025  
Ariel Tyson & Phil Akagu-Jones  
