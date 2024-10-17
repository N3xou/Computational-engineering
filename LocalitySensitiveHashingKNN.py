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


def lsh_knn_cosine(X_train, Y_train, X_test, ks, num_hashes=10, num_buckets=100):
    predictions = {k: [] for k in ks}
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
    input_dim = X_train.shape[1]
    lsh = LSH(num_hashes=num_hashes, num_buckets=num_buckets, input_dim=input_dim)
    lsh.fit(X_train, Y_train)

    for x in X_test:
        candidates = lsh.query(x)
        if len(candidates) == 0:
            continue

        candidates = np.random.choice(candidates, 1000, replace=False)

        candidate_X = np.array([c[0] for c in candidates])
        candidate_y = np.array([c[1] for c in candidates])
        dists = 1 - np.dot(x, candidate_X.T)
        sortedIdx = np.argsort(dists)

        for k in ks:
            closest = sortedIdx[:k]
            nearest_labels = candidate_y[closest]
            prediction = np.bincount(nearest_labels).argmax()
            predictions[k].append(prediction)

    return predictions


def leave_one_out_error(train_X, train_Y, ks, _lsh=False):
    def lsh_leave_one_out_error(train_X, train_Y, ks, num_hashes, num_buckets):
        preds_dict = {k: [] for k in ks}
        for i in range(len(train_X)):

            train_X_loo = np.delete(train_X, i, axis=0)
            train_Y_loo = np.delete(train_Y, i)
            test_X = train_X[i].reshape(1, -1)
            lsh.fit(train_X_loo, train_Y_loo)
            # test_Y = train_Y[i]
            preds = lsh_knn_cosine(train_X_loo, train_Y_loo, test_X, ks, num_hashes=num_hashes, num_buckets=num_buckets)

            for k in ks:
                preds_dict[k].append(preds[k][0])

        return err_rates(preds_dict, train_Y)

    def lsh_plot_error_rate_vs_k(train_X, train_Y, num_hashes=10, num_buckets=100):
        ks = range(1, 4)  # Experiment with K values from 1 to 19
        errors = lsh_leave_one_out_error(train_X, train_Y, ks, num_hashes=num_hashes, num_buckets=num_buckets)

        plt.figure(figsize=(10, 6))
        plt.plot(ks, list(errors.values()), marker='o')
        plt.title("Leave-One-Out Error Rate vs K")
        plt.xlabel("Number of Neighbors (K)")
        plt.ylabel("Leave-One-Out Error Rate")
        plt.grid(True)
        plt.xticks(ks)
        plt.show()

    def err_rates(preds, test_Y):
        ret = {}
        for k, preds_k in preds.items():
            # TODO: fill in error count computation
            ret[k] = np.sum(preds_k != test_Y) / len(test_Y)
        return ret


def plot_error_rate_vs_training_size(train_X, train_Y, k=15, repetitions=100):
    sizes = np.linspace(0.1, 1.0, 20)  # Training sizes from 10% to 100%
    errors = []

    for size in sizes:
        size_errors = []
        for _ in range(repetitions):
            # random sample
            idx = np.random.choice(range(len(train_X)), size=int(size * len(train_X)), replace=False)
            sampled_train_X = train_X[idx]
            sampled_train_Y = train_Y[idx]

            error = leave_one_out_error(sampled_train_X, sampled_train_Y, [k])
            size_errors.append(error[k])

        errors.append(np.mean(size_errors))

    plt.figure(figsize=(10, 6))
    plt.plot(sizes * 100, errors, marker='o')
    plt.title(f"Leave-One-Out Error Rate vs Training Set Size (K={k})")
    plt.xlabel("Training Set Size (%)")
    plt.ylabel("Leave-One-Out Error Rate")
    plt.grid(True)
    plt.xticks(np.arange(10, 101, 10))
    plt.show()


def plot_accuracy(acc_lsh, acc_knn):
    ks = list(acc_lsh.keys())  # Get the k values
    num_k = len(ks)

    # Create an array for x-axis positions
    x = np.arange(num_k)

    # Create subplots for accuracy
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot accuracy for LSH
    ax.bar(x - 0.1, [acc_lsh[k] for k in ks], width=0.2, label='Accuracy LSH', color='blue')

    # Plot accuracy for standard kNN
    ax.bar(x + 0.1, [acc_knn] * num_k, width=0.2, label='Accuracy kNN', color='orange')

    # Labeling
    ax.set_xlabel('k Values')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison: LSH-based kNN vs Standard kNN')
    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.legend()

    # Display the accuracy plot
    plt.tight_layout()
    plt.show()


def plot_runtime(knn_time, lsh_time):
    models = ['Standard kNN', 'LSH-based kNN']
    runtimes = [knn_time, lsh_time]

    # Create subplots for runtime
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot runtime for both models
    ax.bar(models, runtimes, color=['orange', 'blue'])

    # Labeling
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Comparison: Standard kNN vs LSH-based kNN')

    # Display the runtime plot
    plt.tight_layout()
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
y_train = np.array(y_train)
y_test = np.array(y_test)
Ks = [1,2]

#print(KNN(X_train,y_train,X_test, Ks))
#print(lsh_knn_cosine(X_train, y_train, X_test, Ks))
#predictions = lsh_knn_cosine(X_train, y_train, X_test, lsh)
#lsh_plot_error_rate_vs_k(X_train, y_train)
acc_lsh, acc_knn, lsh_time, knn_time, err_lsh,err_knn = measure_speedup_and_accuracy(X_train[:4000], y_train[:4000], X_test, y_test, ks = Ks)
plot_accuracy(acc_lsh, acc_knn)
plot_accuracy(err_lsh,err_knn)
plot_runtime(knn_time, lsh_time)
