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


def knn_cosine(X_train, Y_train, X_test, k=3):
    predictions = []
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
    dists = 1 - np.dot(X_test, X_train.T)
    closest = np.argsort(dists, axis=1)[:, :k]
    predictions = np.array([np.bincount(Y_train[closest[i]]).argmax() for i in range(X_test.shape[0])])

    return predictions

class LSH:
    def __init__(self, num_hashes, num_buckets, input_dim):
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.hash_functions = [np.random.randn(input_dim) for _ in range(num_hashes)]
        self.buckets = {}

    def _hash(self, x):
        return tuple((np.dot(x, h) > 0).astype(int) for h in self.hash_functions)

    def fit(self, X, y):
        for i, x in enumerate(X):
            bucket_id = self._hash(x)
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = []
            self.buckets[bucket_id].append((x, y[i]))

    def query(self, x):
        bucket_id = self._hash(x)
        return self.buckets.get(bucket_id, [])