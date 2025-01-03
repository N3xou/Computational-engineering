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

    # List layers in the model
    print("Feature layers")
    print("--------------")
    for name, layer in zip(vgg.feature_names, vgg.features):
        print("{1: <12} {0: <8}  ({2}".format(name, *str(layer).split("(", 1)))
    print("\nClassifier layers")
    print("-----------------")
    for layer in vgg.classifier:
        print("{: <12}({}".format(*str(layer).split("(", 1)))

ilsvrc = ILSVRC2014Sample(40)
vgg.eval()

figsize(16, 10)
for i in range(10):
    img = ilsvrc.data[30 + i]
    label = ilsvrc.labels[30 + i]

    img_torch = to_tensor(img)
    predicted_label_id = to_np(vgg.predict(img_torch))[0]
    predicted_label = ilsvrc.id_to_label[predicted_label_id]

    desc = ilsvrc.label_to_desc[label].split(",")
    if label == predicted_label:
        desc.append("Classified correctly :)")
    else:
        desc.append("Misclassified as:")
        desc.extend(ilsvrc.label_to_desc[predicted_label].split(","))

    ax = subplot(2, 5, 1 + i)
    ax.set_xlabel("\n".join(desc))  # , {'verticalalignment': 'bottom'})
    ax.set_xticklabels([], visible=False)
    ax.set_yticklabels([], visible=False)
    ax.tick_params(axis="both", which="both", bottom="off", left="off", top="off")
    ax.grid(False)
    imshow(img


import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# List of image paths to classify
images = [
    '/content/im1.jpeg',
    '/content/im2.jpg',
    '/content/im3.jfif',
    '/content/im4.jfif',
    '/content/im5.PNG'
]

top_k = 5  # Number of top predictions to retrieve

# Transformation: resize to 224x224 and convert to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the pre-trained VGG model and set it to evaluation mode
vgg = VGG(model="vgg19")
vgg.eval()
if CUDA:
    vgg.cuda()

# Process each image and get the top-5 predictions
for img_path in images:
    # Load image and apply transformations
    img = PIL.Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    if CUDA:
        img_tensor = img_tensor.cuda()

    # Get probabilities for each class
    probs = vgg.probabilities(img_tensor)

    # Get the top-5 predictions and their corresponding probabilities
    top_probs, top_classes = torch.topk(probs, top_k)

    print(f"Top-{top_k} predictions for {img_path}:")
    for i in range(top_k):
        class_id = top_classes[0][i].item()
        prob = top_probs[0][i].item()
        class_desc = ilsvrc.id_to_desc[class_id]
        print(f"Class: {class_desc}, Probability: {prob:.4f}")
    print("\n")

def obscured_imgs(img, boxsize=8, bsz=64, stride=4):
    h, w, _ = img.shape
    for y in range(0, h - boxsize + 1, stride):
        for x in range(0, w - boxsize + 1, stride):
            img_copy = img.copy()
            img_copy[y:y+boxsize, x:x+boxsize] = 0

            yield img_copy

# Show samples from an obscured batch
obscured_batch = list(obscured_imgs(ilsvrc.data[27], boxsize=56, bsz=12, stride=28))

# Sprawdzanie, czy batch zawiera wystarczającą liczbę obrazów
if len(obscured_batch) < 12:
    print("Warning: Less than expected number of images in the batch")

# Wybierz pierwsze `bsz` obrazy, aby wypełnić batch
batch = obscured_batch[:12]
batch = np.stack(batch)  # Stosujemy np.stack zamiast np.vstack

# Transpozycja dla funkcji plot_mat
batch = batch.transpose(0, 3, 1, 2)

# Wizualizacja obrazu
common.plotting.plot_mat(batch)
# Zaktualizowana funkcja discrete_cmap dla Matplotlib 3.7+
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.colormaps.get_cmap(base_cmap)  # Zaktualizowana składnia
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

# Określenie indeksu obrazu i przypisanie etykiety
idx = 32
img = ilsvrc.data[idx]
label = ilsvrc.labels[idx]

# Parametry okna
bsz = 64
boxsize = 52
stride = 14

vgg.eval()

# Inicjalizacja słowników do przechowywania wyników
map_types = ["heat", "prob", "pred"]
maps = {mt: [] for mt in map_types}

for batch_img in obscured_imgs(img, boxsize, bsz, stride):
    with torch.no_grad():
        batch_tensor = to_tensor(batch_img).unsqueeze(0)
        if CUDA:
            batch_tensor = batch_tensor.cuda()

        # Obliczanie aktywacji dla wybranej warstwy
        heat = vgg.layer_activations(batch_tensor, "conv2_1")[:, 1].sum(dim=(1, 2))
        maps["heat"].append(to_np(heat))

        # Obliczanie prawdopodobieństwa dla poprawnej klasy
        prob = vgg.probabilities(batch_tensor)[:, ilsvrc.label_to_id[label]]
        maps["prob"].append(to_np(prob))

        # Obliczanie predykcji sieci
        pred = torch.argmax(vgg.probabilities(batch_tensor), dim=1)
        maps["pred"].append(to_np(pred))

# Połączenie wyników i zmiana rozmiaru
for k in maps:
    maps[k] = np.concatenate(maps[k])
    maps[k] = maps[k].reshape(int(np.sqrt(len(maps[k]))), -1)

# Rysowanie wykresów
figure(figsize=(16, 10))

# Oryginalny obraz
subplot(2, 2, 1)
imshow(img)
grid(False)
title("Original Image")

# Mapa aktywacji
subplot(2, 2, 2)
imshow(maps["heat"], cmap="viridis")
grid(False)
colorbar()
title("Magnitude of activations in a selected layer\nAs a function of the obscuring square location")

# Mapa prawdopodobieństwa dla poprawnej klasy
subplot(2, 2, 3)
imshow(maps["prob"], cmap="viridis")
grid(False)
colorbar()
title("Probability of the correct class as a function\nof the obscuring square location.")

# Mapa predykcji sieci
subplot(2, 2, 4)
discrete = np.zeros_like(maps["pred"], dtype=np.int64)
values = np.unique(maps["pred"]).astype(np.int64)
for i, v in enumerate(values):
    discrete[np.where(maps["pred"] == v)] = i
num_values = max(values.shape[0], 1)
imshow(discrete, cmap=discrete_cmap(num_values, "cubehelix"))
grid(False)
cb = colorbar(
    ticks=np.linspace(0.5, num_values - 0.5, num_values)  # Automatycznie dostosowane ticki
)
cb.set_ticklabels([ilsvrc.id_to_desc[v] for v in values])
title("The most probable class as a function\nof the obscuring square location")

# Wybór obrazu i wstępne ustawienia
idx = 16
img = ilsvrc.data[idx][None, ...]  # Dodajemy wymiar batch na początku
label = ilsvrc.labels[idx]

predicted_id = to_np(vgg.predict(to_tensor(img)))[0]

print(
    "Predicted: %s (correct: %s)"
    % (ilsvrc.id_to_desc[predicted_id], ilsvrc.label_to_desc[label])
)

num_integration_points = 300
batch_size = 20

assert (num_integration_points % batch_size) == 0

# Generowanie wag interpolacji
weights = np.linspace(0.0, 1.0, num_integration_points).astype("float32")[:, None, None, None]
grads = []

for i in range(0, num_integration_points, batch_size):
    # Tworzenie batch interpolacji pomiędzy szarym obrazem a obrazem docelowym
    batch = weights[i : i + batch_size] * img + (1.0 - weights[i : i + batch_size]) * (np.zeros_like(img) + 0.5)

    # Konwersja batch do tensora z wymogiem gradientu
    batch = to_tensor(batch, requires_grad=True)

    # Obliczenie prawdopodobieństwa dla poprawnej klasy
    probs = vgg.probabilities(batch)[:, predicted_id]
    prediction = torch.sum(probs)

    # Propagacja wsteczna w kierunku wejścia
    prediction.backward()
    grads.append(to_np(batch.grad))

# Konkatenujemy gradienty
grads = np.concatenate(grads, axis=0)
importance_scores = np.abs(grads).mean(axis=0)
importance_scores = importance_scores.squeeze()  # (224, 224)

# Wizualizacja wyników
figsize(16, 10)

# Wykres ważności pikseli
subplot(1, 3, 1)
imshow(importance_scores, cmap="viridis")
colorbar(shrink=0.4)
title("Computed importance of pixels")
grid(False)

# Obraz ważony przez ważność pikseli
subplot(1, 3, 2)
w = np.clip(importance_scores / np.max(np.abs(importance_scores)), 0, 1)  # Normalizacja do [0, 1]
w = np.power(w, 0.7)  # Korekcja gamma

# Zamiast stack, użyj w bezpośrednio, ponieważ ma już 3 kanały
# w = np.stack([w] * 3, axis=-1)  # Upewnij się, że `w` ma wymiar (224, 224, 3)

# Spłaszczenie `img` do wymiaru 3-kanałowego (RGB)
img_rgb = img[0]  # Teraz (224, 224, 3) powinien mieć odpowiednie wymiary

# Upewnij się, że `img_rgb` jest w zakresie [0, 1] lub [0, 255]
weighted_img = np.clip(img_rgb * w, 0, 1)  # Ogranicz wartości do zakresu [0, 1] dla RGB

plt.imshow(weighted_img)
title("Image weighted by the importance")
grid(False)

# Oryginalny obraz
subplot(1, 3, 3)
plt.imshow(np.clip(img_rgb, 0, 1))  # Sprawdzenie zakresu [0, 1]
title("Original image")
grid(False)


![ -e images.tar.bz2 ] || gdown 'https://drive.google.com/uc?id=1qyaQnWVjhpT6L2bf9dRRVtmYdyN9f5pQ' -O images.tar.bz2
![ -d images ] || tar jxf images.tar.bz2    !ls images    arab_text.jpg	   golden_gate.jpg	     shipwreck.jpg	      tubingen.jpg
brad_pitt.jpg	   hoovertowernight.jpg      starry_night_crop.png    woman-with-hat-matisse.jpg
crowd.jpg	   picasso_selfport1907.jpg  starry_night_google.jpg  wood.jpeg
escher_sphere.jpg  pl_text.jpg		     starry_night.jpg
frida_kahlo.jpg    seated-nude.jpg	     the_scream.jpg
 def load_image(path, size=256):
    img = PIL.Image.open(path)
    img = img.convert("RGB")
    img = torchvision.transforms.Resize(size)(img)
    return numpy.asarray(img).astype("float32") / 255.0


imshow(load_image("images/starry_night.jpg"))
grid(False)


class StyleTransferNet(object):
    def __init__(
        self,
        style_layers,
        content_layers,
        vgg_model="vgg19",
        content_weight=5e0,
        style_weight=1e2,
    ):
        self.style_layers = style_layers
        self.style_captures = []
        self.style_weight = style_weight
        self.content_layers = content_layers
        self.content_captures = []
        self.content_weight = content_weight
        self.vgg = VGG(vgg_model)
        self.vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.mse = nn.MSELoss()

    def capture_content(self, x):
        # TODO: implement saving naural activations for a chosen set of content layers
        self.content_captures = TODO

    def capture_style(self, x):
        style_layers = self.vgg.multi_layer_activations(x, self.style_layers)
        self.style_captures = [self._gram_mat(l).detach() for l in style_layers]

    def get_loss(self, x):
        needed_layers = self.content_layers + self.style_layers
        needed_values = self.vgg.multi_layer_activations(x, needed_layers)
        content_values = needed_values[: len(self.content_layers)]
        style_values = needed_values[len(self.content_layers) :]

        self.content_losses = []
        self.style_losses = []

        # TODO:
        # append onto the lists  self.content_losses and self.style_losses
        # the L2 (euclidean) distances between the captured statistics and the
        # reconstructed image's ones. Remember about applying the gram matrix to style!
        # You can use the self.mse to compute the L2 distance

        loss = sum(self.content_losses) * self.content_weight
        loss = loss + sum(self.style_losses) * self.style_weight

        return loss

    def _gram_mat(self, x):
        """
        Compute the matrix of feature correlations.

        Input:
            x: tensor of size batch_size x num_channels x width x heigth

        Output:
            tensor of size batch_size x num_chanels x num_channels given by the formula

            out[b, c1, c2] = 1/width 1/heigth \sum_w=1^width \sum_h=1^heigth x[b, c1, w, h] * x[b, c2, w, h]
        """
        # TODO
        # Compute the Gram matrix, as defined in the docstring. Try to implement it as efficiently as possible.
        pass

style_layers = ["0", "5", "10", "19", "28"]
content_layers = ["21"]
model = StyleTransferNet(
    style_layers=style_layers,
    content_layers=content_layers,
    content_weight=10e0,  # Weight for content loss
    style_weight=1e2      # Weight for style loss
)
if CUDA:
    model.vgg.cuda()

import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import resize

# Funkcja konwertująca obraz na tensor
def to_tensor(img):
    """
    Converts a NumPy array or PIL Image to a PyTorch tensor.
    """
    if isinstance(img, np.ndarray):
        if img.ndim == 2:  # If the input is 2D, assume single channel
            img = img[np.newaxis, :, :]  # Add channel dimension
        return torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    elif isinstance(img, Image.Image):
        img = np.array(img)
        if img.ndim == 2:  # Handle grayscale PIL images
            img = img[np.newaxis, :, :]
        return torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
# Funkcja ładująca obraz z pliku i dopasowująca wymiary
def load_image(image_path, size):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = resize(img, (size, size))  # Dopasowanie rozmiaru obrazu
    return img

# Ścieżki do obrazów stylu i treści
style_img_path = "images/starry_night.jpg"
content_img_path = "images/golden_gate.jpg"

style_img_size = 256
content_img_size = 256

# Ładowanie i przetwarzanie obrazów
style_image = load_image(style_img_path, style_img_size)
content_image = load_image(content_img_path, content_img_size)

# Konwersja obrazów na tensory i przeniesienie na urządzenie (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
style_tensor = to_tensor(style_image).to(device)
content_tensor = to_tensor(content_image).to(device)

# Debugowanie: Wyświetlanie rozmiarów tensorów
print(f"Style Tensor Shape: {style_tensor.shape}, Device: {style_tensor.device}")
print(f"Content Tensor Shape: {content_tensor.shape}, Device: {content_tensor.device}")

# Funkcja przechwytująca zawartość
class Model:
    def __init__(self, device):
        # Definiowanie warstw do przechwycenia (przykład)
        self.device = device
        self.content_layers = [torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1).to(device)]  # Przenosimy warstwę na urządzenie
        # Inne warstwy można dodać do tej listy

    def capture_content(self, x):
        """
        Capture content features from the model.
        """
        print(f"Input Tensor Shape: {x.shape}")  # Debug: Wyświetlanie rozmiaru wejścia
        content_activations = []

        for layer in self.content_layers:
            x = layer(x)  # Zastosowanie warstwy
            content_activations.append(x)

        # Debug: Wyświetlanie aktywacji
        for idx, activation in enumerate(content_activations):
            print(f"Content Layer {idx}: Type = {type(activation)}, Shape = {activation.shape}")

        return content_activations

    def capture_style(self, x):
        """
        Capture style features from the model.
        """
        # Logika dla stylu może być podobna do tego, co robimy dla zawartości
        pass

# Inicjalizacja modelu z odpowiednim urządzeniem
model = Model(device)

# Przechwytywanie zawartości i stylu
print("Capturing content")
content_activations = model.capture_content(content_tensor)

print("Capturing style")
style_activations = model.capture_style(style_tensor)


content_shape = (3, 224, 224)  # Example: RGB image of size 224x224
image = np.random.randn(*content_shape).astype(np.float32) * 0.01 + 0.5
image = to_tensor(image)  # Convert to a PyTorch tensor
image = image.to(device)  # Move to the same device as the model
image.requires_grad = True  # Enable gradient computation for optimization

# Try using Adam or LBFGS optimizer
# optimizer = torch.optim.Adam([image], lr=0.001)  # Uncomment to use Adam
optimizer = torch.optim.LBFGS([image])


import numpy as np
import torch
from matplotlib import pyplot as plt
from IPython import display

# Define `to_np` function
def to_np(tensor):
    """
    Convert PyTorch tensor to NumPy array compatible with Matplotlib.
    """
    if tensor.ndim == 4:  # Handle batched tensors (e.g., (B, C, H, W))
        tensor = tensor[0]  # Remove batch dimension
    tensor = tensor.permute(1, 2, 0).cpu().detach().numpy()  # Convert (C, H, W) to (H, W, C)
    return tensor

# Define variables (replace placeholders with actual data)
content_image = np.random.rand(224, 224, 3)  # Example content image
style_image = np.random.rand(224, 224, 3)    # Example style image
image = torch.rand(1, 3, 224, 224).to("cuda")  # Initial image tensor on GPU (adjust size/device as needed)
image.requires_grad = True

# Placeholder model and optimizer
class DummyModel:
    def __call__(self, x):
        return torch.tensor(0.0, requires_grad=True)  # Dummy loss

    content_losses = [torch.tensor(0.0)]  # Dummy losses
    style_losses = [torch.tensor(0.0)]   # Dummy losses
    content_weight = 1.0
    style_weight = 1.0

model = DummyModel()
optimizer = torch.optim.LBFGS([image])

# Training loop parameters
max_iters = 500
check_every = 5

try:
    # Create the plot
    fig = plt.figure(figsize=(16, 10))

    # Display content and style images
    fig.add_subplot(132).imshow(content_image)
    plt.grid(False)
    fig.add_subplot(133).imshow(style_image)
    plt.grid(False)
    ax = fig.add_subplot(131)
    plt.tight_layout()

    # Initialize the starting image visualization
    imsh = ax.imshow(np.clip(to_np(image), 0.0, 1.0))
    plt.grid(False)

    # Main training loop
    for iter_ in range(max_iters):
        def closure():
            optimizer.zero_grad()
            loss = model(image)  # Compute the loss using the model
            loss.backward()
            return loss

        # Perform optimization step
        loss = optimizer.step(closure)
        # Optional: Clamp image pixel values to valid range
        image.data.clamp_(0, 1)

        # Update and display progress
        if iter_ % check_every == 0:
            display.clear_output(wait=True)
            imsh.set_data(np.clip(to_np(image), 0.0, 1.0))
            display.display(fig)

            print(f"Iteration {iter_: >6} | loss {loss.item(): >10.7f}")
            for i, l in enumerate(model.content_losses):
                print(f"  Content loss {i + 1}: {l.item() * model.content_weight: >8.4f}")
            for i, l in enumerate(model.style_losses):
                print(f"  Style loss {i + 1}: {l.item() * model.style_weight: >10.4f}")

except KeyboardInterrupt:
    display.clear_output(wait=True)
    print(f"Training interrupted at iteration {iter_: >6}. Last loss: {loss.item(): >10.7f}")

plt.imshow(np.clip(to_np(image), 0.0, 1.0))
grid(False)