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