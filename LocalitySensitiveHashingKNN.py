import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

mnist = fetch_openml('mnist_784')
X = normalize(mnist.data)
y = mnist.target.astype(int)


# cosine_distance(a,b)=1−cosine_similarity(a,b)
def cosine_distance(a, b):
    return 1 - (a @ b)


def knn_cosine(X_train, y_train, X_test, k=3):
    predictions = []
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
    dists = 1 - np.dot(X_test, X_train.T)
    closest = np.argsort(cos_dist, axis=1)[:, :k]
    predictions = np.array([np.bincount(y_train[closest[i]]).argmax() for i in range(X_test.shape[0])])

    return predictions