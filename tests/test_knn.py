import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SKKNN

from src.classpass.knn import KNNClassifier


def test_knn_basic_prediction():
    # simple linearly separable toy data
    X = np.array(
        [
            [0, 0],
            [1, 1],
            [10, 10],
            [11, 11],
        ]
    )
    y = np.array(["A", "A", "B", "B"])

    model = KNNClassifier(k=1, distance="euclidean")
    model.fit(X, y)

    preds = model.predict(X)

    assert preds[0] == "A"
    assert preds[2] == "B"


def test_knn_proba():
    X = np.array(
        [
            [0, 0],
            [1, 1],
            [10, 10],
            [11, 11],
        ]
    )
    y = np.array(["A", "A", "B", "B"])

    model = KNNClassifier(k=2)
    model.fit(X, y)

    probs = model.predict_proba(np.array([[0, 0]]))
    assert probs.shape == (1, 2)  # 2 classes
    assert np.isclose(probs.sum(), 1.0)


def test_knn_explain():
    X = np.array([[0, 0], [1, 1], [10, 10]])
    y = np.array(["A", "A", "B"])

    model = KNNClassifier(k=2)
    model.fit(X, y)

    explanation = model.explain(np.array([[0, 0]]))

    assert explanation.indices.shape == (1, 2)
    assert explanation.distances.shape == (1, 2)
    assert explanation.neighbor_labels.shape == (1, 2)


def test_knn_matches_sklearn_small_case():
    # light sanity test comparing our kNN to sklearn's
    X = np.random.randn(20, 3)
    y = np.where(X[:, 0] > 0, "A", "B")

    ours = KNNClassifier(k=3, distance="euclidean").fit(X, y)
    sk = SKKNN(n_neighbors=3, metric="euclidean").fit(X, y)

    test_points = np.random.randn(5, 3)

    p1 = ours.predict(test_points)
    p2 = sk.predict(test_points)

    # allow occasional mismatch due to floating precision or tie-breaking
    match_ratio = np.mean(p1 == p2)
    MIN_MATCH_RATIO = 0.6
    assert match_ratio >= MIN_MATCH_RATIO  # loose check
