# 📘 **ClassPass – Early Student Risk Detection (Custom ML Pipeline + Bayesian Network)**  

**ClassPass** is a fully reproducible, interpretable machine learning and probabilistic AI system for early dropout risk detection in higher education programs.  

It now implements the following AI paradigms:

- **Custom k-Nearest Neighbours classifier**  
- **Custom Decision Tree classifier**  
- **Bayesian Network for probabilistic inference**  

The system uses the UCI *Predict Students’ Dropout and Academic Success* dataset and maps the original 3-class target into a binary label:

- **At Risk** (Dropout)  
- **Continue** (Enrolled + Graduate)

Everything is implemented from first principles to demonstrate clear ML fundamentals, interpretability, and reproducibility.

---

# 🚀 **Features**

## 🔧 Custom ML Models

### **kNN (from scratch)**
- Euclidean / Manhattan distance  
- Predict 
- Neighbour explanations  

### **Decision Tree (from scratch)**
- Entropy or Gini impurity  
- Information gain  
- Customizable max depth  
- Rule extraction  
- Predict 

---

## 🧠 Bayesian Network 

A simple, interpretable Bayesian Network modelling dropout risk using:

```
LowGrades        \\
FinancialRisk ----> DropoutRisk
LowEngagement    //
```

Capabilities:
- Inference via enumeration  
- CPTs learned from the dataset  
- Produces dropout probabilities    
- Complements ML models for comparison  
---



# 🧰 **Installation**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dataset goes in:

```
data/raw/students.csv
```

---

# 🔎 **1. Run EDA**

```bash
python -m scripts.run_eda \
  --data data/raw/students.csv \
  --target Target \
  --binary \
  --outdir reports/eda
```

Outputs:
- `eda_summary.json`  
- `target_counts.csv`

---

# 🤖 **2. Train kNN Baseline**

```bash
python -m scripts.train_baseline \
  --data data/raw/students.csv \
  --target Target \
  --binary \
  --scaler standard \
  --distance euclidean \
  --k-grid 3,5,7,9,11 \
  --outdir reports/figures
```

Outputs:
- `cm_knn.png`  
- `f1_vs_k.png`  
- `artifacts_knn.json`  

---

# 🌳 **3. Train Decision Tree**

```bash
python -m scripts.run_tree \
  --data data/raw/students.csv \
  --target Target \
  --binary \
  --criterion entropy \
  --depth-grid 3,5,7,9 \
  --outdir reports/figures_tree
```

Outputs:
- `cm_tree.png`  
- `tree_rules.txt`  
- `tree_artifacts.json`  

---

# 🧠 **4. Bayesian Network**

Run:

```bash
python -m scripts.run_bn \
  --data data/raw/students.csv \
  --target Target \
  --binary
```

Outputs:
- Bayesian Network metrics printed to console  

Example output:

```
[Bayesian Network Results]
  accuracy: X.XX
  f1_binary: X.XX
  f1_macro: X.XX
```

The BN is intentionally simple to highlight probabilistic reasoning and improve interpretability.

---

# 🧪 **5. Run Tests**

```bash
pytest -q
```

Expected:

```
11 passed in X.XXs
```

---
# 📁 **Project Structure**

```
class-pass/
│
├── src/classpass/
│   ├── data.py
│   ├── preprocessing.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── bayesian_network.py      
│   └── evaluation.py
│
├── scripts/
│   ├── run_eda.py
│   ├── train_baseline.py
│   ├── run_tree.py
│   └── run_bn.py                
│
├── tests/
│   ├── test_eda.py
│   ├── test_knn.py
│   ├── test_decision_tree.py
│   └── test_bayesian_network.py
│
├── data/raw/students.csv
├── reports/
│   ├── eda/
│   ├── figures/
│   └── figures_tree/
│
├── README.md
├── requirements.txt
├── pyproject.toml
└── pytest.ini
```

---

# 📄 **License**

MIT License © 2025  
Ariel Tyson & Phil Akagu-Jones