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
#!pip install idx2numpy

#!mkdir -p ./data/MNIST/raw
#!wget -O ./data/MNIST/raw/train-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
#!wget -O ./data/MNIST/raw/train-labels-idx1-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
#!wget -O ./data/MNIST/raw/t10k-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
#!wget -O ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

#!gunzip ./data/MNIST/raw/*.gz


import idx2numpy
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Normalize, Compose

def load_data(image_path, label_path):
    images = idx2numpy.convert_from_file(image_path).astype('float32')
    labels = idx2numpy.convert_from_file(label_path).astype('int64')
    return torch.tensor(images).unsqueeze(1), torch.tensor(labels)

train_images_path = './data/MNIST/raw/train-images-idx3-ubyte'
train_labels_path = './data/MNIST/raw/train-labels-idx1-ubyte'
test_images_path = './data/MNIST/raw/t10k-images-idx3-ubyte'
test_labels_path = './data/MNIST/raw/t10k-labels-idx1-ubyte'

train_images, train_labels = load_data(train_images_path, train_labels_path)
test_images, test_labels = load_data(test_images_path, test_labels_path)

normalize = Normalize((0.1307,), (0.3081,))
train_images = normalize(train_images)
test_images = normalize(test_images)

train_data = TensorDataset(train_images[:50000], train_labels[:50000])
valid_data = TensorDataset(train_images[50000:], train_labels[50000:])
test_data = TensorDataset(test_images, test_labels)

batch_size = 128
mnist_loaders = {
    "train": DataLoader(train_data, batch_size=batch_size, shuffle=True),
    "valid": DataLoader(valid_data, batch_size=batch_size, shuffle=False),
    "test": DataLoader(test_data, batch_size=batch_size, shuffle=False),
}

def SGD(
    model,
    data_loaders,
    alpha=1e-4,
    epsilon=0.0,
    decay=0.0,
    num_epochs=1,
    max_num_epochs=np.nan,
    patience_expansion=1.5,
    log_every=100,
    device="cpu",
):
    # Put the model in train mode, and move to the evaluation device.
    model.train()
    model.to(device)
    for data_loader in data_loaders.values():
        if isinstance(data_loader, InMemDataLoader):
            data_loader.to(device)

    # Initialize momentum variables
    velocities = [torch.zeros_like(p, device=device) for p in model.parameters()]

    iter_ = 0
    epoch = 0
    best_params = None
    best_val_err = np.inf
    history = {"train_losses": [], "train_errs": [], "val_errs": []}
    print("Training the model!")
    print("Interrupt at any time to evaluate the best validation model so far.")
    try:
        tstart = time.time()
        siter = iter_
        while epoch < num_epochs:
            model.train()
            epoch += 1
            if epoch > max_num_epochs:
                break

            # Learning rate control (schedule per epoch)
            # Example: Exponential decay
            alpha_epoch = alpha * (0.95 ** epoch)

            for x, y in data_loaders["train"]:
                x = x.to(device)
                y = y.to(device)
                iter_ += 1
                out = model(x)
                loss = model.loss(out, y)
                loss.backward()
                _, predictions = out.max(dim=1)
                batch_err_rate = (predictions != y).sum().item() / out.size(0)

                history["train_losses"].append(loss.item())
                history["train_errs"].append(batch_err_rate)

                with torch.no_grad():
                    for (name, p), v in zip(model.named_parameters(), velocities):
                        if "weight" in name:
                            # Weight decay (L2 regularization)
                            p.grad += decay * p

                        # Learning rate schedule per iteration
                        # Example: Cosine annealing based on iteration count
                        alpha_iter = alpha_epoch * (0.5 * (1 + np.cos(np.pi * iter_ / (num_epochs * len(data_loaders["train"])))))

                        # Momentum schedule (if needed)
                        # Example: Linearly increase epsilon up to a max value during the first few epochs
                        epsilon_iter = min(epsilon, epsilon * iter_ / (len(data_loaders["train"]) * 5))

                        # Velocity updates for momentum
                        v[...] = epsilon_iter * v - alpha_iter * p.grad

                        # Update parameters
                        p += v

                        # Zero gradients for the next iteration
                        p.grad.zero_()

                if iter_ % log_every == 0:
                    num_iter = iter_ - siter + 1
                    print(
                        "Minibatch {0: >6}  | loss {1: >5.2f} | err rate {2: >5.2f}%, steps/s {3: >5.2f}".format(
                            iter_,
                            loss.item(),
                            batch_err_rate * 100.0,
                            num_iter / (time.time() - tstart),
                        )
                    )
                    tstart = time.time()

            val_err_rate = compute_error_rate(model, data_loaders["valid"], device)
            history["val_errs"].append((iter_, val_err_rate))

            if val_err_rate < best_val_err:
                # Adjust num of epochs
                num_epochs = int(np.maximum(num_epochs, epoch * patience_expansion + 1))
                best_epoch = epoch
                best_val_err = val_err_rate
                best_params = [p.detach().cpu() for p in model.parameters()]
            clear_output(True)
            m = "After epoch {0: >2} | valid err rate: {1: >5.2f}% | doing {2: >3} epochs".format(
                epoch, val_err_rate * 100.0, num_epochs
            )
            print("{0}\n{1}\n{0}".format("-" * len(m), m))

    except KeyboardInterrupt:
        pass

    if best_params is not None:
        print("\nLoading best params on validation set (epoch %d)\n" % (best_epoch))
        with torch.no_grad():
            for param, best_param in zip(model.parameters(), best_params):
                param[...] = best_param
    plot_history(history)

    class Model(nn.Module):
        def __init__(self, *args, **kwargs):
            super(Model, self).__init__()
            self.layers = nn.Sequential(*args, **kwargs)

        def forward(self, X):
            X = X.view(X.size(0), -1)
            return self.layers.forward(X)

        def loss(self, Out, Targets):
            return F.cross_entropy(Out, Targets)

    model = Model(nn.Linear(28 * 28, 10))

    with torch.no_grad():
        # Initialize parameters
        for name, p in model.named_parameters():
            if "weight" in name:
                p.normal_(0, 0.5)
            elif "bias" in name:
                p.zero_()
            else:
                raise ValueError('Unknown parameter name "%s"' % name)

    # On GPU enabled devices set device='cuda' else set device='cpu'
    t_start = time.time()
    SGD(model, mnist_loaders, alpha=1e-1, max_num_epochs=30, device="cuda")

    test_err_rate = compute_error_rate(model, mnist_loaders["test"])
    m = (
        f"Test error rate: {test_err_rate * 100.0:.3f}%, "
        f"training took {time.time() - t_start:.0f}s."
    )
    print("{0}\n{1}\n{0}".format("-" * len(m), m))

#problem 1
import numpy as np
import torch
import time

def measure_time(func, *args, iterations=10, **kwargs):
    start = time.time()
    for _ in range(iterations):
        func(*args, **kwargs)
    end = time.time()
    return (end - start) / iterations

def matrix_multiplication_loops(A, B):
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i, j] += A[i, k] * B[k, j]
    return result

def matrix_multiplication_einsum(A, B):
    return np.einsum('ik,kj->ij', A, B)

def compare_speeds():
    matrix_shapes = [(100, 100), (200, 200), (500, 500)]
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for shape in matrix_shapes:
        print(f"\nMatrix shape: {shape}")
        A = np.random.rand(*shape).astype(np.float32)
        B = np.random.rand(*shape).astype(np.float32)

        # Python loops
        time_loops = measure_time(matrix_multiplication_loops, A, B, iterations=1)
        print(f"Python loops: {time_loops:.6f} seconds")

        # NumPy einsum
        time_einsum = measure_time(matrix_multiplication_einsum, A, B)
        print(f"NumPy einsum: {time_einsum:.6f} seconds")

        # NumPy dot (CPU)
        time_numpy = measure_time(np.dot, A, B)
        print(f"NumPy dot: {time_numpy:.6f} seconds")

        # PyTorch (CPU and GPU)
        for device in devices:
            A_torch = torch.tensor(A, device=device)
            B_torch = torch.tensor(B, device=device)

            # PyTorch
            time_torch = measure_time(torch.matmul, A_torch, B_torch)
            print(f"PyTorch ({device}): {time_torch:.6f} seconds")

def matrix_transposition_variants():
    size = 500
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)

    A_torch = torch.tensor(A)
    B_torch = torch.tensor(B)
    if torch.cuda.is_available():
        A_torch_gpu = A_torch.to('cuda')
        B_torch_gpu = B_torch.to('cuda')

    variants = {
        "AB": (A_torch, B_torch),
        "A^T B": (A_torch.T, B_torch),
        "A B^T": (A_torch, B_torch.T),
        "A^T B^T": (A_torch.T, B_torch.T),
    }

    print("\nMatrix multiplication variants:")
    for name, (a, b) in variants.items():
        time_cpu = measure_time(torch.matmul, a, b)
        print(f"{name} (CPU): {time_cpu:.6f} seconds")

        if torch.cuda.is_available():
            time_gpu = measure_time(torch.matmul, a.to('cuda'), b.to('cuda'))
            print(f"{name} (GPU): {time_gpu:.6f} seconds")

if __name__ == "__main__":
    print("Comparing speeds of different matrix multiplication methods...")
    compare_speeds()

    print("\nComparing matrix multiplication variants...")
    matrix_transposition_variants()

    # problem 3
    !pip
    install
    tensorflow

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.callbacks import LearningRateScheduler
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    num_layers = 3
    neurons_per_layer = [256, 128, 64]
    initial_lr = 0.05
    momentum = 0.9


    def lr_schedule(epoch):
        if epoch < 10:
            return initial_lr
        elif epoch < 20:
            return initial_lr / 2
        else:
            return initial_lr / 10


    model = Sequential([
        Flatten(input_shape=(28, 28))
    ])

    for neurons in neurons_per_layer:
        model.add(Dense(neurons, activation='relu', kernel_initializer='he_normal'))

    model.add(Dense(10, activation='softmax'))

    optimizer = SGD(learning_rate=initial_lr, momentum=momentum)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    lr_callback = LearningRateScheduler(lr_schedule)

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        callbacks=[lr_callback],
        verbose=1
    )

    mport
    torch.nn as nn
    import torch.nn.functional as F


    class CNNWithDropout(nn.Module):
        def __init__(self, dropout_prob=0.5):
            super(CNNWithDropout, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc1 = nn.Linear(64 * 5 * 5, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(dropout_prob)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
            import torch.optim as optim


    import torch.nn.functional as F


    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                      f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
              f" ({accuracy:.0f}%)\n")
        return accuracy


    model = CNNWithDropout(dropout_prob=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 11):
        train(model, device, mnist_loaders["train"], optimizer, epoch)
        accuracy = test(model, device, mnist_loaders["test"])
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as T

transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNWithDropout(dropout_prob=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.0f}%)\n')
    return accuracy

for epoch in range(1, 11):
    train(model, device, train_loader, optimizer, epoch)
    accuracy = test(model, device, test_loader)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as T

class CNNWithBatchNorm(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(CNNWithBatchNorm, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = F.relu(self.bn2(F.max_pool2d(self.conv2(x), 2)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = CNNWithBatchNorm(dropout_prob=0.5).to(device)

for epoch in range(1, 11):
    train(model, device, train_loader, optimizer, epoch)
    accuracy = test(model, device, test_loader)
