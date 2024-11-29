# Torch has to go first due to an importing bug
import collections
import logging
import os
import re

import httpimport
import numpy as np
import PIL
import scipy.io
import scipy.ndimage

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.autograd import Variable

logging.getLogger("PIL").setLevel(logging.INFO)

#with httpimport.github_repo("janchorowski", "nn_assignments"):
#    import common.plotting

"""
This function contains code to images and weight matrices.
"""

import numpy as np
from matplotlib import pyplot

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

# We strongly recommend training using CUDA
CUDA = True


def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, requires_grad=False):
    x = torch.from_numpy(x)
    if CUDA:
        x = x.cuda()
    # return torch.tensor(x, **kwargs)
    if requires_grad:
        return x.clone().contiguous().detach().requires_grad_(True)
    else:
        return x.clone().contiguous().detach()

![ -e ilsvrc_subsample.tar.bz2 ] || gdown 'https://drive.google.com/uc?id=1C_gmdHnZgSBvOdnfwSi-U-7NixhzZWZA' -O ilsvrc_subsample.tar.bz2
![ -d ilsvrc_subsample ] || tar jxf ilsvrc_subsample.tar.bz2

class ILSVRC2014Sample(object):
    """Mapper from numerical class IDs to their string LABELS and DESCRIPTIONS.

    Please use the dicts:
    - id_to_label and label_to_id to convert string labels and numerical ids
    - label_to_desc to get a textual description of a class label
    - id_to_desc to directly get descriptions for numerical IDs

    """

    def load_image(self, path):
        img = PIL.Image.open(path)
        img = img.convert("RGB")
        for t in self.transforms:
            img = t(img)
        return numpy.asarray(img).astype("float32") / 255.0

    def __init__(self, num=100):
        self.transforms = [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
        ]

        base_dir = "ilsvrc_subsample/"
        devkit_dir = base_dir

        meta = scipy.io.loadmat(devkit_dir + "/meta.mat")
        imagenet_class_names = []
        self.label_to_desc = {}
        for i in range(1000):
            self.label_to_desc[meta["synsets"][i][0][1][0]] = meta["synsets"][i][0][2][
                0
            ]
            imagenet_class_names.append(meta["synsets"][i][0][1][0])

        img_names = sorted(os.listdir(base_dir + "/img"))[:num]
        img_ids = {int(re.search("\d{8}", name).group()) for name in img_names}
        with open(devkit_dir + "/ILSVRC2012_validation_ground_truth.txt", "r") as f:
            self.labels = [
                imagenet_class_names[int(line.strip()) - 1]
                for i, line in enumerate(f)
                if i + 1 in img_ids
            ]
        self.data = [self.load_image(base_dir + "/img/" + name) for name in img_names]

        self.id_to_label = sorted(self.label_to_desc.keys())
        self.label_to_id = {}
        self.id_to_desc = []
        for id_, label in enumerate(self.id_to_label):
            self.label_to_id[label] = id_
            self.id_to_desc.append(self.label_to_desc[label])

def probabilities(self, x):
    """Return class probabilities."""
    logits = self(x)
    return self.softmax(logits)

def predict(self, x):
    """Return predicted class IDs."""
    probs = self.probabilities(x)
    return torch.argmax(probs, dim=1)


class VGGPreprocess(torch.nn.Module):
    """Pytorch module that normalizes data for a VGG network
    """

    # These values are taken from http://pytorch.org/docs/master/torchvision/models.html
    RGB_MEANS = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
    RGB_STDS = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

    def forward(self, x):
        """Normalize a single image or a batch of images

        Args:
            x: a pytorch Variable containing and float32 RGB image tensor with
              dimensions (batch_size x width x heigth x RGB_channels) or
              (width x heigth x RGB_channels).
        Returns:
            a torch Variable containing a normalized BGR image with shape
              (batch_size x BGR_channels x width x heigth)
        """
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        # x is batch * width * heigth *channels,
        # make it batch * channels * width * heigth
        if x.size(3) == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        means = self.RGB_MEANS
        stds = self.RGB_STDS
        if x.is_cuda:
            means = means.cuda()
            stds = stds.cuda()
        x = (x - Variable(means)) / Variable(stds)
        return x


class VGG(torch.nn.Module):
    """Wrapper around a VGG network allowing convenient extraction of layer activations."""

    FEATURE_LAYER_NAMES = {
        "vgg16": [
            "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
            "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
            "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "pool3",
            "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "pool4",
            "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "pool5",
        ],
        "vgg19": [
            "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
            "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
            "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "conv3_4", "relu3_4", "pool3",
            "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "conv4_4", "relu4_4", "pool4",
            "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "conv5_4", "relu5_4", "pool5",
        ],
    }

    def __init__(self, model="vgg19"):
        super(VGG, self).__init__()
        all_models = {
            "vgg16": torchvision.models.vgg16,
            "vgg19": torchvision.models.vgg19,
        }
        vgg = all_models[model](pretrained=True)

        self.preprocess = VGGPreprocess()
        self.features = vgg.features
        self.classifier = vgg.classifier
        self.softmax = torch.nn.Softmax(dim=-1)
        self.feature_names = self.FEATURE_LAYER_NAMES[model]

        assert len(self.feature_names) == len(self.features)

    def forward(self, x):
        """Return pre-softmax unnormalized logits."""
        x = self.preprocess(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def probabilities(self, x):
        """Return class probabilities."""
        logits = self(x)
        return self.softmax(logits)

    def layer_activations(self, x, layer_name):
        """Return activations of a selected layer."""
        x = self.preprocess(x)
        for name, layer in zip(self.feature_names, self.features):
            x = layer(x)
            if name == layer_name:
                return x
        raise ValueError("Layer %s not found" % layer_name)

    def multi_layer_activations(self, x, layer_names):
        """Return activations of all requested layers."""
        pass  # TODO implement me

    def predict(self, x):
        """Return predicted class IDs."""
        probs = self.probabilities(x)
        return torch.argmax(probs, dim=1)

vgg = VGG("vgg19")

if CUDA:
    vgg.cuda()