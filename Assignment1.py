#!pip install -q gdown httpimport

#![ -e mnist.npz ] || gdown 'https://drive.google.com/uc?id=1X_OMH9qBGfzpLMumdRtHXzvHWPYWJEsm' -O mnist.npz

# Standard IPython notebook imports
%matplotlib inline

import os

import httpimport
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy.stats as sstats

import seaborn as sns
from sklearn import datasets

# In this way we can import functions straight from gitlab
#with httpimport.gitlab_repo('SHassonaProjekt', 'inzynieria_obliczeniowa_23_24'):
#     from common.plotting import plot_mat

#sns.set_style('whitegrid')

def scale_mat(mat, lower=0., upper=1.):
    """
    Scale all linearly all elements in a mtrix into a given range.
    """
    ret = mat - np.min(mat)
    return ret * ((upper-lower) / np.max(ret)) + lower

def get_grid(num_elem, prop=(9,16)):
    """
    Find grid proportions that would accomodate given number of elements.
    """
    cols = np.ceil(np.sqrt(1. * num_elem * prop[1] / prop[0]))
    rows = np.ceil(1. * num_elem / cols)
    while cols != np.ceil(1. * num_elem / rows):
        cols = np.ceil(1. * num_elem / rows)
        rows = np.ceil(1. * num_elem / cols)
    return int(rows), int(cols)

def plot_mat(mat, scaleIndividual=True, colorbar=False, prop=(9,16), gutters=2,
             scale_fun=scale_mat, **kwargs):
    """
    Plot an image for each entry in the tensor.

    Inputs
    ------
    mat: 4D tensor, n_images x n_channels x rows x columns

    """
    nSamples, nChannels, r, c = mat.shape
    gr, gc =  get_grid(nSamples, (prop[0]*c, prop[1]*r))
    toPlot = np.zeros((int(gr*r+(gr-1)*gutters), int(gc*c + (gc-1)*gutters), nChannels) ) + np.NaN
    for s in range(nSamples):
        pr = s // gc
        pc = s - (pr*gc)
        small_img = mat[s,:,:,:].transpose(1,2,0)
        if scaleIndividual:
            small_img = scale_fun(small_img)
        toPlot[int(pr*(r+gutters)):int(pr*(r+gutters)+r),
               int(pc*(c+gutters)):int(pc*(c+gutters)+c),:] = small_img
    if nChannels==1:
        pyplot.imshow(toPlot[:,:,0], interpolation='nearest', **kwargs)
    else:
        pyplot.imshow(toPlot, interpolation='nearest', **kwargs)
    if colorbar:
        pyplot.colorbar()
    pyplot.axis('off')

# PROBLEM 1

# Task 1

a = 10
b = 2.5 * 10**23
c = 2 + 3j
d = np.exp(2j * np.pi / 3)

# Task 2

aVec = [3.14,15,9,26]
bVec = np.arange(5,-5,-0.2)
cVec = np.logspace(0, 1, 10) # co rozumiemy przez correct length?
dVec = "Hello"
print(cVec)

# Task 3

aMat = 2 * np.ones((9, 9))
bMat = np.zeros((9, 9)) + np.diag([1, 2, 3, 4, 5, 4, 3, 2, 1])
#print(bMat)
cMat = np.arange(1, 101).reshape((10, 10), order='F') # F - fortean, reshaping column wise
#print(cMat)
dMat = np.zeros((3, 4)) + np.nan
#print(dMat)
eMat = np.array([[13,-1,5], [-22,10,-87]])
#print(eMat)
fMat = np.floor((np.random.rand(3, 3) * 7 - 3)) # *6 tworzy range [0:7] -3 przesuwa o 3 w lewo [-3:4] , floor zaokragla w dol wiec liczby wychodza -3:3
# rand generuje [0, 1) , nizszy zakres wlacznie, wyzszy wylacznie
print(fMat)

# Task 4

mulMat = np.outer(np.arange(1,11), np.arange(1,11))
print(mulMat)

# Task 5

xVec = 1/(np.sqrt(2*np.pi*(2.5)**2)*np.exp(-cVec**2/(2*np.pi*(2.5)**2)))
print(xVec)
yVec = np.log10(1 / cVec)
print(yVec)

# Task 6

# t = transpose, zamiana rzedu na kolumne w tym przypadku
row = np.array([[0,1,2,3,4,5,6]])
print(row)
column = np.array([[0], [10], [20], [30], [40], [50], [60]])
print(column)
xMat = row @ column
print(xMat)
yMat = column @ row
print(yMat)

# Task 7

def ismagic(A):
    row_sum = np.sum(A[0,:])
    if A.shape[0] != A.shape[1]:
        return False
    for i in range(A.shape[0]):
        if np.sum(A[i,:]) != row_sum:
            return False
        elif np.sum(A[:,i]) != row_sum:
            return False
    return True


assert not ismagic(np.array([[1,1], [2,2]]))
assert ismagic(np.array([[2,7,6],[9,5,1],[4,3,8]]))

# PROBLEM 2

iris_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# Use read_csv to load the data. Make sure you get 150 examples!
iris_df = pd.read_csv(iris_url)


# Set the column names to
# 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

# Print the first 10 entries
iris_df.head(10)

# Show numerical summary of the data, using DataFrame.describe()
iris_df.describe()

# Plot the data using seaborn's pairplot
sns.pairplot(iris_df)

#melt
iris_df_long = pd.melt(iris_df,
                       id_vars='target',
                       value_vars='sepal_length',
                       var_name='variable',
                       value_name='value')
iris_df_long.head()

# plots

# Hint: use a `catplot`
sns.catplot(kind='box', data=iris_df, x='target', y='sepal_length', hue='target')

# TODO: create two more plots, using a boxenplot and a swarmplot.

sns.catplot(kind='boxen', data=iris_df, x='target', y='sepal_length', hue='target')
sns.catplot(kind='swarm', data=iris_df, x='target', y='sepal_length', hue='target')

unknown_df = pd.DataFrame(
    [[1.5, 0.3, 'unknown'],
     [4.5, 1.2, 'unknown'],
     [5.1, 1.7, 'unknown'],
     [5.5, 2.3, 'unknown']],
     columns=['petal_length', 'petal_width', 'target'])

sns.scatterplot(x='petal_length', y='petal_width', hue='target', data=iris_df)
sns.scatterplot(x='petal_length', y='petal_width', color='gray', marker='v',
                label='unknown', s=70, data=unknown_df)

# PRoblem 3, KNN implementation

from scipy import stats
def KNN(train_X, train_Y, test_X, ks, verbose=False):
    """
    Compute predictions for various k
    Args:
        train_X: array of shape Ntrain x D
        train_Y: array of shape Ntrain
        test_X: array of shape Ntest x D
        ks: list of integers
    Returns:
        preds: dict k: predictions for k
    """
    # Cats data to float32
    train_X = train_X.astype(np.float32)
    test_X = test_X.astype(np.float32)

    # Alloc space for results
    preds = {}

    if verbose:
        print("Computing distances... ", end='')
    #
    # TODO: fill in an efficient distance matrix computation
    #
    dists = np.sqrt(np.sum((test_X[:, None] - train_X) ** 2, axis=2))


    if verbose:
        print("Sorting... ", end='')

    # TODO: findes closest trainig points
    # Hint: use argsort
    closest = dists.argsort(axis=1)
    if verbose:
        print("Computing predictions...", end='')

    targets = train_Y[closest]

    for k in ks:
        k_closest = targets[:, :k]
        predictions = []
        # print(closest)
        for labels in k_closest:
            unique_labels, counts = np.unique(labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]
            predictions.append(most_common_label)

        preds[k] = np.array(predictions)
    if verbose:
        print("Done")
    return preds

# Now classify the 4 unknown points
iris_x = np.array(iris_df[['petal_length', 'petal_width']])
iris_y = np.array(iris_df['target'])

unknown_x = np.array(unknown_df[['petal_length', 'petal_width']])

KNN(iris_x, iris_y, unknown_x, [1, 3, 5, 7], verbose=True)
#########
iris_x = np.array(iris_df[['petal_length', 'petal_width']])
iris_y = np.array(iris_df['target'])

#print(iris_x)
# ? mesh_x, mesh_y = np.meshgrid(iris_x,iris_y)

#use np.unique with suitable options to map the class names to numbers
target_names, iris_y_ids = np.unique(iris_y, return_inverse=True)
mesh_x, mesh_y = np.meshgrid(iris_x,iris_y_ids)
#print(target_names)
#print(iris_y_ids)
mesh_data = np.hstack([mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1)])
#print(mesh_data)
preds = KNN(iris_x, iris_y_ids, mesh_data, [1, 3, 5, 7])
for k, preds_k in preds.items():
    plt.figure()
    plt.title(f"Decision boundary for k={k}")
    plt.contourf(mesh_x, mesh_y, preds_k.reshape(mesh_x.shape),alpha=0.5, c = 'iris_y_ids')
    plt.scatter(iris_x[:, 0], iris_x[:, 1], c=iris_y_ids, edgecolor='k', cmap=plt.cm.RdYlBu)

#
#TODO: write a function to compute error rates
def err_rates(preds, test_Y):
    ret = {}
    for k, preds_k in  preds.items():
        # TODO: fill in error count computation
        ret[k] = np.sum(preds_k != test_Y) / len(test_Y)
    return ret

#
iris_x = np.array(iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
iris_y = np.array(iris_df['target'])

ks = range(1, 30, 2)
results = []

for _rep in tqdm(range(1000)):
    #TODO
    # Use np.random.randint to get training indices
    # The find all unselected indices to form a test set
    train_idx = ...
    test_idx = ...

    #TODO: apply your kNN classifier to data subset
    preds = KNN(...)
    errs = err_rates(preds, iris_y[test_idx])

    for k, errs_k in errs.items():
        results.append({'K':k, 'err_rate': errs_k})

# results_df will be a data_frame in long format
results_df = pd.DataFrame(results)

plt.figure()
sns.regplot(...)

# KNN testing

iris_x = np.array(iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
iris_y = np.array(iris_df['target'])

ks = range(1, 30, 2)
results = []

for _rep in tqdm(range(1000)):
    #TODO
    # Use np.random.randint to get training indices
    # The find all unselected indices to form a test set
    train_idx = np.random.randint(0, len(iris_x), len(iris_x))
    test_idx = np.array([i for i in range(len(iris_x)) if i not in train_idx])

    #TODO: apply your kNN classifier to data subset
    preds = KNN(iris_x, iris_y, test_idx ,ks)
    errs = err_rates(preds, iris_y[test_idx])

    for k, errs_k in errs.items():
        results.append({'K':k, 'err_rate': errs_k})

# results_df will be a data_frame in long format
results_df = pd.DataFrame(results)

plt.figure()
sns.regplot(results_df)

# todo: KNN ? contouf ?