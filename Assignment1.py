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
