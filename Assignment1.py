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
#sns.catplot(...)
