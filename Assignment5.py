#%matplotlib inline
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torchvision.datasets
from torch import nn

def compute_error_rate(model, data_loader, device="cpu"):
    """Evaluate model on all samples from the data loader.
    """
    # Put the model in eval mode, and move to the evaluation device.
    model.eval()
    model.to(device)
    if isinstance(data_loader, InMemDataLoader):
        data_loader.to(device)

    num_errs = 0.0
    num_examples = 0
    # we don't need gradient during eval!
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model.forward(x)
            _, predictions = outputs.data.max(dim=1)
            num_errs += (predictions != y.data).sum().item()
            num_examples += x.size(0)
    return num_errs / num_examples


def plot_history(history):
    """Helper to plot the trainig progress over time."""
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    train_loss = np.array(history["train_losses"])
    plt.semilogy(np.arange(train_loss.shape[0]), train_loss, label="batch train loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    train_errs = np.array(history["train_errs"])
    plt.plot(np.arange(train_errs.shape[0]), train_errs, label="batch train error rate")
    val_errs = np.array(history["val_errs"])
    plt.plot(val_errs[:, 0], val_errs[:, 1], label="validation error rate", color="r")
    plt.ylim(0, 0.20)
    plt.legend()

    class InMemDataLoader(object):
        """
        A data loader that keeps all data in CPU or GPU memory.
        """

        __initialized = False

        def __init__(
                self,
                dataset,
                batch_size=1,
                shuffle=False,
                sampler=None,
                batch_sampler=None,
                drop_last=False,
        ):
            """A torch dataloader that fetches data from memory."""
            batches = []
            for i in tqdm(range(len(dataset))):
                batch = [torch.tensor(t) for t in dataset[i]]
                batches.append(batch)
            tensors = [torch.stack(ts) for ts in zip(*batches)]
            dataset = torch.utils.data.TensorDataset(*tensors)
            self.dataset = dataset
            self.batch_size = batch_size
            self.drop_last = drop_last

            if batch_sampler is not None:
                if batch_size > 1 or shuffle or sampler is not None or drop_last:
                    raise ValueError(
                        "batch_sampler option is mutually exclusive "
                        "with batch_size, shuffle, sampler, and "
                        "drop_last"
                    )
                self.batch_size = None
                self.drop_last = None

            if sampler is not None and shuffle:
                raise ValueError("sampler option is mutually exclusive with " "shuffle")

            if batch_sampler is None:
                if sampler is None:
                    if shuffle:
                        sampler = torch.utils.data.RandomSampler(dataset)
                    else:
                        sampler = torch.utils.data.SequentialSampler(dataset)
                batch_sampler = torch.utils.data.BatchSampler(
                    sampler, batch_size, drop_last
                )

            self.sampler = sampler
            self.batch_sampler = batch_sampler
            self.__initialized = True

        def __setattr__(self, attr, val):
            if self.__initialized and attr in ("batch_size", "sampler", "drop_last"):
                raise ValueError(
                    "{} attribute should not be set after {} is "
                    "initialized".format(attr, self.__class__.__name__)
                )

            super(InMemDataLoader, self).__setattr__(attr, val)

        def __iter__(self):
            for batch_indices in self.batch_sampler:
                yield self.dataset[batch_indices]

        def __len__(self):
            return len(self.batch_sampler)

        def to(self, device):
            self.dataset.tensors = tuple(t.to(device) for t in self.dataset.tensors)
            return self

# Monkey-patch MNIST to use a more robust MIST mirror
torchvision.datasets.MNIST.resources = [
    (
        "https://web.archive.org/web/20150906081542/http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    ),
    (
        "https://web.archive.org/web/20150906081542/http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "d53e105ee54ea40749a09fcbcd1e9432",
    ),
    (
        "https://web.archive.org/web/20150906081542/http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "9fb629c4189551a2d022fa330f9573f3",
    ),
    (
        "https://web.archive.org/web/20150906081542/http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        "ec29112dd5afa0611ce80d1b7f02629c",
    ),
]