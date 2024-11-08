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