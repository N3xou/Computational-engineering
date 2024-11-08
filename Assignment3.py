!pip install -q gdown httpimport

# Standard IPython notebook imports
%matplotlib inline

import os
from io import StringIO

import graphviz
import httpimport
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sstats
import seaborn as sns
import sklearn.ensemble
import sklearn.tree
from tqdm import tqdm_notebook

# In this way we can import functions straight from gitlab
with httpimport.gitlab_repo('SHassonaProjekt', 'inzynieria_obliczeniowa_23_24'):
     from common.plotting import plot_mat

sns.set_style("whitegrid")

columns = [
    "target",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises?",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

# Use read_csv to load the data.
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
mushroom_df = pd.read_csv(url, header=None, names=columns)
mushroom_idx_df = mushroom_df.reset_index()

# 2. Iris
iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris_df = pd.read_csv(
    iris_url,
    header=None,
    names=["sepal_length", "sepal_width", "petal_length", "petal_width", "target"],
)

# 3. Congressoinal Voting
# Binary attributes, binary class, missing data
vote_df = (
    pd.read_csv(
        "https://pkgstore.datahub.io/machine-learning/vote/vote_csv/data/65f1736301dee4a2ad032abfe2a61acb/vote_csv.csv"
    )
    .rename({"Class": "target"}, axis=1)
    .fillna("na")
)

# 4. Adult
# census records, continuous and categorical attributes (some ordered), missing values
adult_names = [
    "Age",
    "Workclass",
    "fnlwgt",
    "Education",
    "Education-Num",
    "Martial Status",
    "Occupation",
    "Relationship",
    "Race",
    "Sex",
    "Capital Gain",
    "Capital Loss",
    "Hours per week",
    "Country",
    "target",
]
adult_df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    names=adult_names,
    header=None,
    na_values="?",
)
adult_test_df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    names=adult_names,
    header=None,
    na_values="?",
    skiprows=1,
)

# 5. German Credit

german_df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
    names=[f"A{d}" for d in range(1, 21)] + ["target"],
    header=None,
    sep=" ",
)

## start

def entropy(series):
    counts = series.value_counts(normalize=True)
    ent = -np.sum(counts * np.log2(counts))
    return ent

mushroom_df.apply(entropy)


def entropy(series):
    """Compute the entropy of a pandas Series."""
    counts = series.value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts))


def cond_entropy(df, X, Y):
    """Compute the conditional entropy H(Y|X) in dataframe df.

    Args:
        df: a dataframe
        X: the name of the conditioning column
        Y: the name of the column whose entropy we wish to compute
    """
    # Group by X and calculate entropy of Y for each group
    grouped = df.groupby(X)[Y]

    # Compute weighted average of entropies
    conditional_entropy = grouped.apply(entropy).multiply(grouped.size() / len(df)).sum()

    return conditional_entropy


# Load the dataset (assuming `mushroom_df` is already loaded)
# Compute conditional entropy H(target | C) for each column C
target_column = 'class'  # Assuming the target is the 'class' column (edible/poisonous label)
conditional_entropies = {col: cond_entropy(mushroom_df, col, target_column)
                         for col in mushroom_df.columns if col != target_column}

# Sort columns by conditional entropy
sorted_cond_entropies = pd.Series(conditional_entropies).sort_values()
print("Conditional Entropies (H(Y|X)):\n", sorted_cond_entropies)
print("Most informative variable:", sorted_cond_entropies.idxmin())


from sklearn.feature_selection import mutual_info_classif

mushroom_df['ID'] = range(len(mushroom_df))


# Target column
X = mushroom_df[['ID']]  # Only using the ID column
y = mushroom_df[target_column]  # Assuming `target_column` is set correctly

# Compute mutual information
mi_id_target = mutual_info_classif(X, y, discrete_features=True)[0]
print("Mutual Information between ID and Target:", mi_id_target)


import numpy as np
import matplotlib.pyplot as plt


def entropy(counts):
    total = sum(counts)
    probabilities = [count / total for count in counts if count > 0]
    return -sum(p * np.log2(p) for p in probabilities)

def gini(counts):
    total = sum(counts)
    probabilities = [count / total for count in counts]
    return 1 - sum(p ** 2 for p in probabilities)

def mean_err_rate(counts):
    total = sum(counts)
    probabilities = [count / total for count in counts]
    return 1 - max(probabilities)

probabilities = np.linspace(0, 1, 100)
entropies = []
ginis = []
mean_err_rates = []

for p in probabilities:
    counts = [p, 1 - p]
    entropies.append(entropy(counts))
    ginis.append(gini(counts))
    mean_err_rates.append(mean_err_rate(counts))

plt.figure(figsize=(12, 6))
plt.plot(probabilities, entropies, label="Entropy", color="blue")
plt.plot(probabilities, ginis, label="Gini Index", color="green")
plt.plot(probabilities, mean_err_rates, label="Mean Error Rate", color="red")
plt.xlabel("Probability of class 1")
plt.ylabel("Purity Measure")
plt.title("Purity Measures for a Two-Class Set")
plt.legend()
plt.show()


# Varying probability for a two-class system
prob_class_1 = np.linspace(0, 1, 100)
entropy_vals = []
gini_vals = []
mean_err_vals = []

# Calculate each purity measure across different probabilities
for p in prob_class_1:
    counts = np.array([p, 1 - p]) * 100  # Simulate counts based on probabilities
    entropy_vals.append(entropy(counts))
    gini_vals.append(gini(counts))
    mean_err_vals.append(mean_err_rate(counts))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(prob_class_1, entropy_vals, label="Entropy")
plt.plot(prob_class_1, gini_vals, label="Gini Impurity")
plt.plot(prob_class_1, mean_err_vals, label="Mean Error Rate")
plt.xlabel("Probability of Class 1")
plt.ylabel("Purity Measure")
plt.title("Purity Measures for a Two-Class System")
plt.legend()
plt.show()


class CategoricalMultivalueSplit(AbstractSplit):
    def build_subtrees(self, df, subtree_kwargs):
        self.subtrees = {}
        for group_name, group_df in df.groupby(self.attr):
            child = Tree(group_df, **subtree_kwargs)
            self.subtrees[group_name] = child

    def __call__(self, x):
        return self.subtrees[x[self.attr]]

    def iter_subtrees(self):
        return self.subtrees.values()

    def add_to_graphviz(self, dot, parent, print_info):
        for split_name, child in self.subtrees.items():
            child.add_to_graphviz(dot, print_info)
            dot.edge(f"{id(parent)}", f"{id(child)}", label=f"{split_name}")

def get_categorical_split_and_purity(
    df, parent_purity, purity_fun, attr, normalize_by_split_entropy=False
):
    """Return a multivariate split and its purity.
    """
    split = CategoricalMultivalueSplit(attr)
    # cal pur
    mean_child_purity = 0
    total_instances = len(df)

    # calc pur for child
    for group_name, group_df in df.groupby(attr):
        prob_group = len(group_df) / total_instances
        child_purity = purity_fun(group_df['target'].value_counts())
        mean_child_purity += prob_group * child_purity

    # pur gain
    purity_gain = parent_purity - mean_child_purity
    if normalize_by_split_entropy:
        purity_gain /= entropy(df[attr].value_counts())
    return split, purity_gain

class NumericalSplit(AbstractSplit):
    def __init__(self, attr, threshold):
        super().__init__(attr)
        self.threshold = threshold

    def build_subtrees(self, df, subtree_kwargs):
        self.subtrees = {}
        left_split = df[df[self.attr] <= self.threshold]
        right_split = df[df[self.attr] > self.threshold]
        self.subtrees['<='] = Tree(left_split, **subtree_kwargs)
        self.subtrees['>'] = Tree(right_split, **subtree_kwargs)

    def __call__(self, x):
        return self.subtrees['<='] if x[self.attr] <= self.threshold else self.subtrees['>']

    def iter_subtrees(self):
        return self.subtrees.values()

    def add_to_graphviz(self, dot, parent, print_info):
        for split_name, child in self.subtrees.items():
            child.add_to_graphviz(dot, print_info)
            dot.edge(f"{id(parent)}", f"{id(child)}", label=f"{split_name} {self.threshold}")

    def __str__(self):
        return f"NumericalSplit({self.attr} <= {self.threshold})"
def get_split(df, criterion="infogain", nattrs=None):
    """Find best split on the given dataframe."""
    target_value_counts = df["target"].value_counts()
    if len(target_value_counts) == 1:
        return None

    possible_splits = [col for col in df.columns if col != "target" and len(df[col].unique()) > 1]
    assert "target" not in possible_splits

    if not possible_splits:
        return None

    if criterion in ["infogain", "infogain_ratio"]:
        purity_fun = entropy
    elif criterion in ["mean_err_rate"]:
        purity_fun = mean_err_rate
    elif criterion in ["gini"]:
        purity_fun = gini
    else:
        raise Exception("Unknown criterion: " + criterion)

    base_purity = purity_fun(target_value_counts)
    best_purity_gain = -1
    best_split = None

    if nattrs is not None:
        possible_splits = np.random.choice(possible_splits, nattrs, replace=False)

    for attr in possible_splits:
        if np.issubdtype(df[attr].dtype, np.number):
            split_sel_fun = get_numerical_split_and_purity
        else:
            split_sel_fun = get_categorical_split_and_purity

        split, purity_gain = split_sel_fun(
            df,
            base_purity,
            purity_fun,
            attr,
            normalize_by_split_entropy=criterion.endswith("ratio"),
        )

        if purity_gain > best_purity_gain:
            best_purity_gain = purity_gain
            best_split = split

    return best_split
class Tree:
    def __init__(self, df, **kwargs):
        super().__init__()
        # Assert that there are no missing values,
        # TODO: remove this for bonus problem #2.4
        assert not df.isnull().values.any()

        # Technicality:
        # We need to let subtrees know about all targets to properly color nodes
        # We pass this in subtree arguments.
        if "all_targets" not in kwargs:
            kwargs["all_targets"] = sorted(df["target"].unique())
        # Save keyword arguments to build subtrees
        kwargs_orig = dict(kwargs)

        # Get kwargs we know about, remaining ones will be used for splitting
        self.all_targets = kwargs.pop("all_targets")

        # Save debug info for visualization
        # Debugging tip: contents of self.info are printed in tree visualizations!
        self.counts = df["target"].value_counts()
        self.info = {
            "num_samples": len(df),
            "entropy": entropy(self.counts),
            "gini": gini(self.counts),
        }

        self.split = get_split(df, **kwargs)
        if self.split:
            self.split.build_subtrees(df, kwargs_orig)

    def get_target_distribution(self, sample):
        """Return the target distribution at the leaf node for the given sample."""
        if self.split is None:
            # Leaf node, return class distribution
            return self.counts
        else:
            # Internal node, descend into the appropriate subtree
            subtree = self.split(sample)
            return subtree.get_target_distribution(sample)

    def classify(self, sample):
        """Classify a sample by returning the most common target class."""
        target_distribution = self.get_target_distribution(sample)
        return target_distribution.idxmax()

    def draw(self, print_info=True):
        dot = graphviz.Digraph()
        self.add_to_graphviz(dot, print_info)
        return dot

    def add_to_graphviz(self, dot, print_info):
        freqs = self.counts / self.counts.sum()
        freqs = dict(freqs)
        colors = []
        freqs_info = []
        for i, c in enumerate(self.all_targets):
            freq = freqs.get(c, 0.0)
            if freq > 0:
                colors.append(f"{i % 9 + 1};{freq}")
                freqs_info.append(f"{c}:{freq:.2f}")
        colors = ":".join(colors)
        labels = [" ".join(freqs_info)]
        if print_info:
            for k, v in self.info.items():
                labels.append(f"{k} = {v}")
        if self.split:
            labels.append(f"split by: {self.split.attr}")
        dot.node(
            f"{id(self)}",
            label="\n".join(labels),
            shape="box",
            style="striped",
            fillcolor=colors,
            colorscheme="set19",
        )
        if self.split:
            self.split.add_to_graphviz(dot, self, print_info)

            class NumericalSplit(AbstractSplit):
                def __init__(self, attr, th):
                    super(NumericalSplit, self).__init__(attr)
                    self.th = th

                def build_subtrees(self, df, subtree_kwargs):
                    self.subtrees = (
                        Tree(df[df[self.attr] <= self.th], **subtree_kwargs),
                        Tree(df[df[self.attr] > self.th], **subtree_kwargs),
                    )

                def __call__(self, x):
                    # Return the subtree for the data sample `x`
                    if x[self.attr] <= self.th:
                        return self.subtrees[0]
                    else:
                        return self.subtrees[1]

                def __str__(self):
                    return f"NumericalSplit: {self.attr} <= {self.th}"

                def iter_subtrees(self):
                    return self.subtrees

                def add_to_graphviz(self, dot, parent, print_info):
                    self.subtrees[0].add_to_graphviz(dot, print_info)
                    dot.edge(f"{id(parent)}", f"{id(self.subtrees[0])}", label=f"<= {self.th:.2f}")
                    self.subtrees[1].add_to_graphviz(dot, print_info)
                    dot.edge(f"{id(parent)}", f"{id(self.subtrees[1])}", label=f"> {self.th:.2f}")

            def get_numrical_split_and_purity(
                    df, parent_purity, purity_fun, attr, normalize_by_split_entropy=False
            ):
                """Find the best split threshold and compute the average purity after a split."""
                attr_df = df[[attr, "target"]].sort_values(attr)
                targets = attr_df["target"]
                values = attr_df[attr]
                right_counts = targets.value_counts()
                left_counts = right_counts * 0

                best_split = None  # Will be None, or NumericalSplit(attr, best_threshold)
                best_purity_gain = -1
                N = len(attr_df)

                for row_i in range(N - 1):
                    row_target = targets.iloc[row_i]
                    attribute_value = values.iloc[row_i]
                    next_attribute_value = values.iloc[row_i + 1]
                    split_threshold = (attribute_value + next_attribute_value) / 2.0

                    # Update left and right counts
                    left_counts[row_target] += 1
                    right_counts[row_target] -= 1

                    if attribute_value == next_attribute_value:
                        continue

                    left_purity = purity_fun(left_counts)
                    right_purity = purity_fun(right_counts)
                    mean_child_purity = (left_purity * left_counts.sum() + right_purity * right_counts.sum()) / N
                    purity_gain = parent_purity - mean_child_purity

                    if purity_gain > best_purity_gain:
                        best_purity_gain = purity_gain
                        best_split = NumericalSplit(attr, split_threshold)

                return best_split, best_purity_gain
iris2d = iris_df[["petal_length", "petal_width", "target"]]

iris2d_tree = Tree(iris2d, criterion="infogain")
iris2d_tree.draw()

mesh_x, mesh_y = np.meshgrid(
    np.linspace(iris2d.petal_length.min(), iris2d.petal_length.max(), 100),
    np.linspace(iris2d.petal_width.min(), iris2d.petal_width.max(), 100),
)

mesh_data = np.hstack([mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1)])
mesh_data = pd.DataFrame(mesh_data, columns=iris2d.columns[:-1])

preds = np.empty((len(mesh_data),))

for criterion in ["infogain", "infogain_ratio", "gini", "mean_err_rate"]:
    iris2d_tree = Tree(iris2d, criterion=criterion)
    for i, (_, r) in enumerate(mesh_data.iterrows()):
        preds[i] = iris2d_tree.all_targets.index(iris2d_tree.classify(r))

    plt.figure()
    plt.title(f"Iris2D decision boundary for {criterion}.")
    plt.contourf(
        mesh_x, mesh_y, preds.reshape(mesh_x.shape), cmap="Set1", vmin=0, vmax=7
    )
    sns.scatterplot(
        x="petal_length", y="petal_width", hue="target", data=iris_df, palette="Set1",
    )
    plt.show()

import pandas as pd
import numpy as np
import graphviz
from collections import defaultdict

def entropy(counts):
    total = counts.sum()
    prob = counts / total
    return -np.sum(prob * np.log2(prob + 1e-9))

def gini(counts):
    total = counts.sum()
    prob = counts / total
    return 1 - np.sum(prob**2)

def mean_err_rate(counts):
    total = counts.sum()
    prob = counts / total
    return 1 - prob.max()

class AbstractSplit:
    def __init__(self, attr):
        self.attr = attr

    def build_subtrees(self, df, subtree_kwargs):
        raise NotImplementedError

    def __call__(self, sample):
        raise NotImplementedError

    def iter_subtrees(self):
        raise NotImplementedError

    def add_to_graphviz(self, dot, parent, print_info):
        raise NotImplementedError

class NumericalSplit(AbstractSplit):
    def __init__(self, attr, th):
        super().__init__(attr)
        self.th = th

    def build_subtrees(self, df, subtree_kwargs):
        self.subtrees = (
            Tree(df[df[self.attr] <= self.th], **subtree_kwargs),
            Tree(df[df[self.attr] > self.th], **subtree_kwargs),
        )

    def __call__(self, x):
        return self.subtrees[0] if x[self.attr] <= self.th else self.subtrees[1]

    def __str__(self):
        return f"NumericalSplit: {self.attr} <= {self.th}"

    def iter_subtrees(self):
        return self.subtrees

    def add_to_graphviz(self, dot, parent, print_info):
        self.subtrees[0].add_to_graphviz(dot, print_info)
        dot.edge(f"{id(parent)}", f"{id(self.subtrees[0])}", label=f"<= {self.th:.2f}")
        self.subtrees[1].add_to_graphviz(dot, print_info)
        dot.edge(f"{id(parent)}", f"{id(self.subtrees[1])}", label=f"> {self.th:.2f}")


class Tree:
    def __init__(self, df, **kwargs):
        super().__init__()

        if "all_targets" not in kwargs:
            kwargs["all_targets"] = sorted(df["target"].unique())

        kwargs_orig = dict(kwargs)
        self.all_targets = kwargs.pop("all_targets")
        self.counts = df["target"].value_counts()
        self.info = {
            "num_samples": len(df),
            "entropy": entropy(self.counts),
            "gini": gini(self.counts),
        }

        self.split = get_split(df, **kwargs)
        if self.split:
            self.split.build_subtrees(df, kwargs_orig)

    def get_target_distribution(self, sample):
        if self.split is None:
            return self.counts
        else:
            if pd.isnull(sample[self.split.attr]):
                left_counts = self.split.subtrees[0].counts.sum()
                right_counts = self.split.subtrees[1].counts.sum()
                total_counts = left_counts + right_counts

                left_prob = left_counts / total_counts
                right_prob = right_counts / total_counts

                left_dist = self.split.subtrees[0].get_target_distribution(sample)
                right_dist = self.split.subtrees[1].get_target_distribution(sample)

                return left_prob * left_dist + right_prob * right_dist
            else:
                return self.split(sample).get_target_distribution(sample)

    def classify(self, sample):
        return self.get_target_distribution(sample).idxmax()

    def draw(self, print_info=True):
        dot = graphviz.Digraph()
        self.add_to_graphviz(dot, print_info)
        return dot

    def add_to_graphviz(self, dot, print_info):
        freqs = self.counts / self.counts.sum()
        freqs = dict(freqs)
        colors = []
        freqs_info = []
        for i, c in enumerate(self.all_targets):
            freq = freqs.get(c, 0.0)
            if freq > 0:
                colors.append(f"{i%9 + 1};{freq}")
                freqs_info.append(f"{c}:{freq:.2f}")
        colors = ":".join(colors)
        labels = [" ".join(freqs_info)]
        if print_info:
            for k, v in self.info.items():
                labels.append(f"{k} = {v}")
        if self.split:
            labels.append(f"split by: {self.split.attr}")
        dot.node(
            f"{id(self)}",
            label="\n".join(labels),
            shape="box",
            style="striped",
            fillcolor=colors,
            colorscheme="set19",
        )
        if self.split:
            self.split.add_to_graphviz(dot, self, print_info)

def get_split(df, criterion="infogain", nattrs=None):
    target_value_counts = df["target"].value_counts()
    if len(target_value_counts) == 1:
        return None

    possible_splits = [col for col in df.columns if col != "target" and len(df[col].unique()) > 1]
    assert "target" not in possible_splits

    if not possible_splits:
        return None

    if criterion in ["infogain", "infogain_ratio"]:
        purity_fun = entropy
    elif criterion in ["mean_err_rate"]:
        purity_fun = mean_err_rate
    elif criterion in ["gini"]:
        purity_fun = gini
    else:
        raise Exception("Unknown criterion: " + criterion)

    base_purity = purity_fun(target_value_counts)
    best_purity_gain = -1
    best_split = None

    if nattrs is not None:
        possible_splits = np.random.choice(possible_splits, nattrs, replace=False)

    for attr in possible_splits:
        if np.issubdtype(df[attr].dtype, np.number):
            split_sel_fun = get_numerical_split_and_purity
        else:
            split_sel_fun = get_categorical_split_and_purity

        split, purity_gain = split_sel_fun(
            df,
            base_purity,
            purity_fun,
            attr,
            normalize_by_split_entropy=criterion.endswith("ratio"),
        )

        if purity_gain > best_purity_gain:
            best_purity_gain = purity_gain
            best_split = split

    return best_split