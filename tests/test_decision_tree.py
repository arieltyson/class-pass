import numpy as np
from src.classpass.decision_tree import DecisionTreeClassifier


def test_tree_basic_split():
    X = np.array([
        [0],
        [1],
        [10],
        [11],
    ])
    y = np.array(["A", "A", "B", "B"])

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds[0] == "A"
    assert preds[2] == "B"


def test_tree_stump():
    X = np.array([[0], [1], [2], [100]])
    y = np.array(["A", "A", "A", "B"])

    model = DecisionTreeClassifier(max_depth=1)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds[-1] == "B"


def test_tree_rules_generate():
    X = np.array([[0], [10]])
    y = np.array(["A", "B"])

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)

    rules = model.extract_rules()
    assert len(rules) >= 1
    assert "Predict" in rules[0]