import numpy as np

from classpass.knn import KNNClassifier


def test_knn_simple():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array(["A", "A", "B", "B"])
    clf = KNNClassifier(k=1).fit(X, y)
    assert clf.predict(np.array([[0, 0]]))[0] == "A"
    assert clf.predict(np.array([[1, 1]]))[0] == "B"
