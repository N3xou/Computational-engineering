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


Please fill the purity measures below.

Verify the correctness by plotting the purity values if a two-class set with given class probabilities


import numpy as np
import matplotlib.pyplot as plt


def entropy(counts):
    """Calculate the entropy of a distribution given class counts."""
    probabilities = counts / np.sum(counts)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Avoid log(0)


def gini(counts):
    """Calculate the Gini impurity of a distribution given class counts."""
    probabilities = counts / np.sum(counts)
    return 1 - np.sum(probabilities ** 2)


def mean_err_rate(counts):
    """Calculate the mean error rate of a distribution given class counts."""
    probabilities = counts / np.sum(counts)
    return 1 - np.max(probabilities)


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
