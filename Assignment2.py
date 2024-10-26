# Please note that this code needs only to be run in a fresh runtime.
# However, it can be rerun afterwards too.
!pip install -q gdown httpimport

# Standard IPython notebook imports
import itertools
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sopt
import scipy.stats as sstats
import seaborn as sns
import sklearn.ensemble
import sklearn.tree
from sklearn import datasets
from tqdm.auto import tqdm

import graphviz
import httpimport

# In this way we can import functions straight from gitlab
with httpimport.gitlab_repo('SHassonaProjekt', 'inzynieria_obliczeniowa_23_24'):
    from common.gradients import check_gradient
    from common.plotting import plot_mat
# Standard IPython notebook imports
import itertools
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sopt
import scipy.stats as sstats
import seaborn as sns
import sklearn.ensemble
import sklearn.tree
from sklearn import datasets
from tqdm.auto import tqdm

import graphviz
import httpimport

# In this way we can import functions straight from gitlab


def scale_mat(mat, lower=0., upper=1.):
    """
    Scale all linearly all elements in a mtrix into a given range.
    """
    ret = mat - np.min(mat)
    return ret * ((upper-lower) / np.max(ret)) + lower


def check_gradient(f, X, delta=1e-4, prec=1e-6):
    fval, fgrad = f(X)
    num_grad = numerical_gradient(f, X, delta=delta)
    diffnorm = np.sqrt(np.sum((fgrad-num_grad)**2))
    gradnorm = np.sqrt(np.sum(fgrad**2))
    if gradnorm>0:
        if not (diffnorm < prec or diffnorm/gradnorm < prec):
            raise Exception("Numerical and anaylical gradients differ: %s != %s!" %
                            (num_grad, fgrad))
    else:
        if not (diffnorm < prec):
            raise Exception("Numerical and anaylical gradients differ: %s != %s!" %
                            (num_grad, fgrad))
    return True
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


sns.set_style("whitegrid")
sns.set_style("whitegrid")

wiki_table = u"""English|French|German|Spanish|Portuguese|Esperanto|Italian|Turkish|Swedish|Polish|Dutch|Danish|Icelandic|Finnish|Czech
a|8.167|7.636|6.516|11.525|14.634|12.117|11.745|12.920|9.383|10.503|7.486|6.025|10.110|12.217|8.421
b|1.492|0.901|1.886|2.215|1.043|0.980|0.927|2.844|1.535|1.740|1.584|2.000|1.043|0.281|0.822
c|2.782|3.260|2.732|4.019|3.882|0.776|4.501|1.463|1.486|3.895|1.242|0.565|0|0.281|0.740
d|4.253|3.669|5.076|5.010|4.992|3.044|3.736|5.206|4.702|3.725|5.933|5.858|1.575|1.043|3.475
e|12.702|14.715|16.396|12.181|12.570|8.995|11.792|9.912|10.149|7.352|18.91|15.453|6.418|7.968|7.562
f|2.228|1.066|1.656|0.692|1.023|1.037|1.153|0.461|2.027|0.143|0.805|2.406|3.013|0.194|0.084
g|2.015|0.866|3.009|1.768|1.303|1.171|1.644|1.253|2.862|1.731|3.403|4.077|4.241|0.392|0.092
h|6.094|0.737|4.577|0.703|0.781|0.384|0.636|1.212|2.090|1.015|2.380|1.621|1.871|1.851|1.356
i|6.966|7.529|6.550|6.247|6.186|10.012|10.143|9.600|5.817|8.328|6.499|6.000|7.578|10.817|6.073
j|0.153|0.613|0.268|0.493|0.397|3.501|0.011|0.034|0.614|1.836|1.46|0.730|1.144|2.042|1.433
k|0.772|0.049|1.417|0.011|0.015|4.163|0.009|5.683|3.140|2.753|2.248|3.395|3.314|4.973|2.894
l|4.025|5.456|3.437|4.967|2.779|6.104|6.510|5.922|5.275|2.564|3.568|5.229|4.532|5.761|3.802
m|2.406|2.968|2.534|3.157|4.738|2.994|2.512|3.752|3.471|2.515|2.213|3.237|4.041|3.202|2.446
n|6.749|7.095|9.776|6.712|4.446|7.955|6.883|7.987|8.542|6.237|10.032|7.240|7.711|8.826|6.468
o|7.507|5.796|2.594|8.683|9.735|8.779|9.832|2.976|4.482|6.667|6.063|4.636|2.166|5.614|6.695
p|1.929|2.521|0.670|2.510|2.523|2.755|3.056|0.886|1.839|2.445|1.57|1.756|0.789|1.842|1.906
q|0.095|1.362|0.018|0.877|1.204|0|0.505|0|0.020|0|0.009|0.007|0|0.013|0.001
r|5.987|6.693|7.003|6.871|6.530|5.914|6.367|7.722|8.431|5.243|6.411|8.956|8.581|2.872|4.799
s|6.327|7.948|7.270|7.977|6.805|6.092|4.981|3.014|6.590|5.224|3.73|5.805|5.630|7.862|5.212
t|9.056|7.244|6.154|4.632|4.336|5.276|5.623|3.314|7.691|2.475|6.79|6.862|4.953|8.750|5.727
u|2.758|6.311|4.166|2.927|3.639|3.183|3.011|3.235|1.919|2.062|1.99|1.979|4.562|5.008|2.160
v|0.978|1.838|0.846|1.138|1.575|1.904|2.097|0.959|2.415|0.012|2.85|2.332|2.437|2.250|5.344
w|2.360|0.074|1.921|0.017|0.037|0|0.033|0|0.142|5.813|1.52|0.069|0|0.094|0.016
x|0.150|0.427|0.034|0.215|0.253|0|0.003|0|0.159|0.004|0.036|0.028|0.046|0.031|0.027
y|1.974|0.128|0.039|1.008|0.006|0|0.020|3.336|0.708|3.206|0.035|0.698|0.900|1.745|1.043
z|0.074|0.326|1.134|0.467|0.470|0.494|1.181|1.500|0.070|4.852|1.39|0.034|0|0.051|1.503
à|0|0.486|0|0|0.072|0|0.635|0|0|0|0|0|0|0|0
â|0|0.051|0|0|0.562|0|0|0|0|0|0|0|0|0|0
á|0|0|0|0.502|0.118|0|0|0|0|0|0|0|1.799|0|0.867
å|0|0|0|0|0|0|0|0|1.338|0|0|1.190|0|0.003|0
ä|0|0|0.578|0|0|0|0|0|1.797|0|0|0|0|3.577|0
ã|0|0|0|0|0.733|0|0|0|0|0|0|0|0|0|0
ą|0|0|0|0|0|0|0|0|0|0.699|0|0|0|0|0
æ|0|0|0|0|0|0|0|0|0|0|0|0.872|0.867|0|0
œ|0|0.018|0|0|0|0|0|0|0|0|0|0|0|0|0
ç|0|0.085|0|0|0.530|0|0|1.156|0|0|0|0|0|0|0
ĉ|0|0|0|0|0|0.657|0|0|0|0|0|0|0|0|0
ć|0|0|0|0|0|0|0|0|0|0.743|0|0|0|0|0
č|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0.462
ď|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0.015
ð|0|0|0|0|0|0|0|0|0|0|0|0|4.393|0|0
è|0|0.271|0|0|0|0|0.263|0|0|0|0|0|0|0|0
é|0|1.504|0|0.433|0.337|0|0|0|0|0|0|0|0.647|0|0.633
ê|0|0.218|0|0|0.450|0|0|0|0|0|0|0|0|0|0
ë|0|0.008|0|0|0|0|0|0|0|0|0|0|0|0|0
ę|0|0|0|0|0|0|0|0|0|1.035|0|0|0|0|0
ě|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1.222
ĝ|0|0|0|0|0|0.691|0|0|0|0|0|0|0|0|0
ğ|0|0|0|0|0|0|0|1.125|0|0|0|0|0|0|0
ĥ|0|0|0|0|0|0.022|0|0|0|0|0|0|0|0|0
î|0|0.045|0|0|0|0|0|0|0|0|0|0|0|0|0
ì|0|0|0|0|0|0|0.030|0|0|0|0|0|0|0|0
í|0|0|0|0.725|0.132|0|0|0|0|0|0|0|1.570|0|1.643
ï|0|0.005|0|0|0|0|0|0|0|0|0|0|0|0|0
ı|0|0|0|0|0|0|0|5.114|0|0|0|0|0|0|0
ĵ|0|0|0|0|0|0.055|0|0|0|0|0|0|0|0|0
ł|0|0|0|0|0|0|0|0|0|2.109|0|0|0|0|0
ñ|0|0|0|0.311|0|0|0|0|0|0|0|0|0|0|0
ń|0|0|0|0|0|0|0|0|0|0.362|0|0|0|0|0
ň|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0.007
ò|0|0|0|0|0|0|0.002|0|0|0|0|0|0|0|0
ö|0|0|0.443|0|0|0|0|0.777|1.305|0|0|0|0.777|0.444|0
ô|0|0.023|0|0|0.635|0|0|0|0|0|0|0|0|0|0
ó|0|0|0|0.827|0.296|0|0|0|0|1.141|0|0|0.994|0|0.024
õ|0|0|0|0|0.040|0|0|0|0|0|0|0|0|0|0
ø|0|0|0|0|0|0|0|0|0|0|0|0.939|0|0|0
ř|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0.380
ŝ|0|0|0|0|0|0.385|0|0|0|0|0|0|0|0|0
ş|0|0|0|0|0|0|0|1.780|0|0|0|0|0|0|0
ś|0|0|0|0|0|0|0|0|0|0.814|0|0|0|0|0
š|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0.688
ß|0|0|0.307|0|0|0|0|0|0|0|0|0|0|0|0
ť|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0.006
þ|0|0|0|0|0|0|0|0|0|0|0|0|1.455|0|0
ù|0|0.058|0|0|0|0|0.166|0|0|0|0|0|0|0|0
ú|0|0|0|0.168|0.207|0|0|0|0|0|0|0|0.613|0|0.045
û|0|0.060|0|0|0|0|0|0|0|0|0|0|0|0|0
ŭ|0|0|0|0|0|0.520|0|0|0|0|0|0|0|0|0
ü|0|0|0.995|0.012|0.026|0|0|1.854|0|0|0|0|0|0|0
ů|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0.204
ý|0|0|0|0|0|0|0|0|0|0|0|0|0.228|0|0.995
ź|0|0|0|0|0|0|0|0|0|0.078|0|0|0|0|0
ż|0|0|0|0|0|0|0|0|0|0.706|0|0|0|0|0
ž|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0.721"""
df = pd.read_table(io.StringIO(wiki_table), sep="|", index_col=0)
df.head()


langs = list(df)
letters = list(df.index)
print("Languages:", ",".join(langs))
print("Letters:", ", ".join(letters))
print("P(ę|Polish) =", df.loc["ę", "Polish"])


# The values are percentages of letter appearance, but curiously enough they don't
# sum to 100%.
print(f"\nTotal letter count by language:\n{df.sum(0)}")
from sklearn import preprocessing
# Thus we normalize the data such that the letter frequencies add up to 1 for each language
df_norm = preprocessing.normalize(df, norm="l1", axis=1)
print(f"\nAfter normalization:\n{df_norm.sum(0)}")

def naive_bayes(sent, langs, df):
    """Returns the most probable language of a sentence"""

    # Try working with log-probabilities.
    # to prevent taking log(0) you can e.g. add a very small amount (1e-100)
    # to each tabulated frequency.
    df_log = np.log(df + 1e-100)

    # normalize the sentence: remove spaces and punctuations, take lower case
    sent = ''.join(char for char in sent if char.isalnum()).lower()

    log_probs = {}
    for lang in langs:
        log_prob = 0
        for letter in sent:
            if letter in df_log.index:
                log_prob += df_log[lang].get(letter, np.log(1e-100))

        log_probs[lang] = log_prob

    # TODO compute language probabilitie and order from most to least probable
    max_log_prob = max(log_probs.values())
    probs = {lang: np.exp(log_prob - max_log_prob) for lang, log_prob in log_probs.items()}

    total_prob = sum(probs.values())
    probs = {lang: prob / total_prob for lang, prob in probs.items()}

    return probs


sentences = [
    "No dejes para mañana lo que puedas hacer hoy.",
    "Przed wyruszeniem w drogę należy zebrać drużynę.",
    "Żeby zrozumieć rekurencję, należy najpierw zrozumieć rekurencję.",
    "Si vale la pena hacerlo vale la pena hacerlo bien.",
    "Experience is what you get when you didn't get what you wanted.",
    "Należy prowokować intelekt, nie intelektualistów.",
]

for sent in sentences:
    print(f"Sentence: '{sent}'")
    probs = naive_bayes(sent, langs, df_norm)

    # Get the most probable language by finding the max probability
    most_probable_lang = max(probs, key=probs.get)
    print(f"Predicted Language: {most_probable_lang}")

    # Print probabilities for additional context

    print("\n")


#
# The true polynomial relation:
# y(x) = 1 + 2x -5x^2 + 4x^3
#
# TODO: write down the proper coefficients
#


def powers_of_X(X, degree):
    powers = np.arange(degree + 1).reshape(1, -1)
    return X ** powers


def compute_polynomial(X, Theta):
    XP = powers_of_X(X, len(Theta) - 1)  # len(Theta) x N
    Y = XP @ Theta
    return Y.reshape(-1, 1)


true_poly_theta = np.array([1.0, 2.0, -5, 4])



def make_dataset(N, theta=true_poly_theta, sigma=0.1):
    """ Sample a dataset """
    X = np.random.uniform(size=(N, 1))
    Y_clean = compute_polynomial(X, theta)
    Y = Y_clean + np.random.randn(N, 1) * sigma
    return X, Y


train_data = make_dataset(30)
XX = np.linspace(0, 1, 100).reshape(-1, 1)
YY = compute_polynomial(XX, true_poly_theta)
plt.scatter(train_data[0], train_data[1], label="train data", color="r")
plt.plot(XX, compute_polynomial(XX, true_poly_theta), label="ground truth")
plt.legend(loc="upper left")

def poly_fit(data, degree, alpha):
    "Fit a polynomial of a given degree and weight decay parameter alpha"
    X = powers_of_X(data[0], degree)
    Y = data[1].reshape(-1, 1)

    N, d = X.shape
    I = np.eye(d)

    Theta =np.linalg.solve(X.T @ X + alpha * I, X.T @ Y)
    return Theta

num_test_samples = 100
num_train_samples = [30]
alphas = [0.0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
degrees = range(15)
num_repetitions = 30


# sample a single test dataset for all experiments
test_data = make_dataset(num_test_samples)
results = []

for (repetition, num_train, alpha, degree,) in itertools.product(
    range(num_repetitions), num_train_samples, alphas, degrees
):
    train_data = make_dataset(num_train)
    Theta = poly_fit(train_data, degree, alpha)
    Y_train_pred = compute_polynomial(train_data[0], Theta)
    Y_test_pred = compute_polynomial(test_data[0], Theta)
    train_err = np.mean((Y_train_pred - train_data[1]) ** 2)
    test_err = np.mean((Y_test_pred - test_data[1]) ** 2)
    results.append(
        {
            "repetition": repetition,
            "num_train": num_train,
            "alpha": alpha,
            "degree": degree,
            "dataset": "train",
            "err_rate": train_err,
        }
    )
    results.append(
        {
            "repetition": repetition,
            "num_train": num_train,
            "alpha": alpha,
            "degree": degree,
            "dataset": "test",
            "err_rate": test_err,
        }
    )
results_df = pd.DataFrame(results)
results_df.head()


# TODO
#
# Plot how the error rates depend on the the polynomial degree and regularization
# constant.
# Try to find the best value for lambda on the test set, explain the model
# behavoir for small lambdas and large lambdas.
#
# Hint: the plots below all use sns.relplot!
#
sns.set(style="whitegrid")

sns.relplot(
    x="degree", y="err_rate", hue="alpha", kind="line", col="dataset",
    data=results_df[results_df["dataset"] == "train"],
    palette="tab10", col_wrap=2, height=4, aspect=1.5
)
plt.suptitle("Training Error by Degree and Alpha")
plt.show()

sns.relplot(
    data=results_df,
    x="degree",
    y="err_rate",
    hue="alpha",
    kind="line"
)

plt.title("Error rates as a function of polynomial degree and alpha")
plt.show()

# TODO
# Now set a small regularizatoin for numerical stability  (e.g. alpha=1e-6)
# and present the relationship between
# train and test error rates for varous degrees of the polynomial for
# different sizes of the train set.
#
train_sample_sizes = [10, 30, 50, 100]
reg_alpha = 1e-6
poly_degrees = range(15)
test_sample_size = 100
repeats = 30

test_dataset = make_dataset(test_sample_size)

error_results = []

for (rep, train_size, degree) in itertools.product(
    range(repeats), train_sample_sizes, poly_degrees
):
    train_dataset = make_dataset(train_size)

    model_params = poly_fit(train_dataset, degree, reg_alpha)

    train_predictions = compute_polynomial(train_dataset[0], model_params)
    test_predictions = compute_polynomial(test_dataset[0], model_params)

    train_error = np.mean((train_predictions - train_dataset[1]) ** 2)
    test_error = np.mean((test_predictions - test_dataset[1]) ** 2)

    error_results.append(
        {
            "rep": rep,
            "train_size": train_size,
            "degree": degree,
            "dataset": "train",
            "err_rate": train_error,
        }
    )
    error_results.append(
        {
            "rep": rep,
            "train_size": train_size,
            "degree": degree,
            "dataset": "test",
            "err_rate": test_error,
        }
    )

error_df = pd.DataFrame(error_results)

sns.relplot(
    data=error_df,
    x="degree",
    y="err_rate",
    hue="dataset",
    col="train_size",
    kind="line",
    facet_kws={'sharey': False},
    markers=True,
)

plt.title("Train vs Test Error for Different Polynomial Degrees and Training Sizes (alpha=1e-6)")
plt.show()


from sklearn.preprocessing import StandardScaler
def generate_dataset(N, sigma=0.1):
    """ Generate dataset with x∝U(0;10) and
        y∝N(μ=1+0.2x−0.05x^2+0.004x^3, σ=0.1) """
    X = np.random.uniform(0, 10, size=(N, 1))
    Y_actual = 1 + 0.2 * X - 0.05 * X**2 + 0.004 * X**3
    Y = Y_actual + np.random.randn(N, 1) * sigma
    return X, Y

# Polynomial fitting with normalization and ridge regularization
def polynomial_fit_with_scaling(data, degree, alpha):
    """Fit polynomial with normalization and ridge regularization."""
    X_poly = expand_features(data[0], degree)
    Y = data[1].reshape(-1, 1)

    # Apply scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    Theta = np.linalg.solve(X_scaled.T @ X_scaled + alpha * np.eye(X_scaled.shape[1]), X_scaled.T @ Y)
    return Theta, scaler

# Model evaluation on train and test datasets
def evaluate_polynomial_model(train_data, test_data, degree, alpha):
    """Evaluate model performance on train and test data."""
    Theta, scaler = polynomial_fit_with_scaling(train_data, degree, alpha)

    # Transform training data
    X_train_scaled = scaler.transform(expand_features(train_data[0], degree))
    train_error = np.mean((X_train_scaled @ Theta - train_data[1]) ** 2)

    # Transform test data
    X_test_scaled = scaler.transform(expand_features(test_data[0], degree))
    test_error = np.mean((X_test_scaled @ Theta - test_data[1]) ** 2)

    return train_error, test_error

# Generate polynomial features (excluding bias term)
def expand_features(X, degree):
    return np.vander(X.flatten(), degree + 1, increasing=True)[:, 1:]

train_sizes = [10, 20, 50, 100, 200]
polynomial_degrees = range(1, 10)
regularization_param = 1e-6
test_set = generate_dataset(100)

results = []

# Iterate over train sizes and polynomial degrees
for size in train_sizes:
    for deg in polynomial_degrees:
        train_set = generate_dataset(size)
        train_error, test_error = evaluate_polynomial_model(train_set, test_set, deg, regularization_param)
        results.append({
            "train_size": size,
            "degree": deg,
            "dataset": "train",
            "error_rate": train_error
        })
        results.append({
            "train_size": size,
            "degree": deg,
            "dataset": "test",
            "error_rate": test_error
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plotting the results
sns.relplot(
    data=results_df,
    x="degree", y="error_rate", hue="dataset", col="train_size", kind="line",
    facet_kws={"sharey": False, "sharex": True}, col_wrap=3
)
plt.show()


#
# Implement the Rosenbrock function
#


def rosenbrock_v(x):
    """Returns the value of Rosenbrock's function at x"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rosenbrock(x):
    """Returns the value of rosenbrock's function and its gradient at x
    """
    val = rosenbrock_v(x)
    # Gradient should be np.array
    dVdX= np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
        200 * (x[1] - x[0]**2)
    ])
    return [val, dVdX]


#
# Feel free to add your own test points.
#
for test_point in [[0.0, 0.0], [1.0, 1.0], [0.5, 1.0], [1.0, 0.5]]:
    assert check_gradient(rosenbrock, np.array(test_point), prec=1e-5)

    lbfsg_hist = []


    def save_hist(x):
        lbfsg_hist.append(np.array(x))


    x_start = [0.0, 2.0]
    lbfsgb_ret = sopt.fmin_l_bfgs_b(rosenbrock, x_start, callback=save_hist)

    # TODO: plot the countours of the function and overlay the optimization trajectory
    x = np.linspace(0, 2, 400)
    y = np.linspace(0, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock_v([X, Y])

    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='viridis')
    plt.colorbar(contour)
    plt.title("Contour plot of the Rosenbrock function (Upper Right Quadrant)")
    plt.xlabel("x")
    plt.ylabel("y")

    hist = np.array(lbfsg_hist)
    plt.plot(hist[:, 0], hist[:, 1], 'r.-', markersize=5, label='Optimization Path')
    plt.plot(lbfsgb_ret[0][0], lbfsgb_ret[0][1], 'bo', label='Optimum')

    plt.xlim(0, 2)
    plt.ylim(0, 3)
    plt.legend()

    import numpy as np
    import matplotlib.pyplot as plt


    def rosenbrock(x):
        """Rosenbrock's function."""
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


    def rosenbrock_grad(x):
        """Gradient of Rosenbrock's function."""
        dfdx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
        dfdx1 = 200 * (x[1] - x[0] ** 2)
        return np.array([dfdx0, dfdx1])


    def rosenbrock_hessian(x):
        """Hessian of Rosenbrock's function."""
        d2fdx0dx0 = 2 - 400 * (x[1] - 3 * x[0] ** 2)
        d2fdx0dx1 = -400 * x[0]
        d2fdx1dx1 = 200
        return np.array([[d2fdx0dx0, d2fdx0dx1], [d2fdx0dx1, d2fdx1dx1]])


    def Newton(f, Theta0, alpha, stop_tolerance=1e-10, max_steps=1000000):
        """Newton-Raphson method with simple line search."""
        Theta = Theta0
        history = [Theta0]
        fun_evals = 0

        for step in range(int(max_steps)):  # Convert max_steps to an integer
            val, grad = f(Theta)[:2]
            hessian = f(Theta)[2]

            # Compute the Newton direction
            try:
                direction = -np.linalg.inv(hessian).dot(grad)
            except np.linalg.LinAlgError:
                # If Hessian is singular, fall back to gradient descent
                direction = -grad

            # Line search: backtracking
            t = 1.0
            while f(Theta + t * direction)[0] > val + 1e-4 * t * grad.dot(direction):
                t *= 0.5
                fun_evals += 1

            # Update the parameters
            Theta = Theta + t * direction
            history.append(Theta)

            # Check convergence
            if np.linalg.norm(grad) < stop_tolerance:
                break

        return Theta, history, fun_evals


    # Initial point
    x_start = np.array([-1.2, 1.0])

    # Run the Newton method
    Xopt, Xhist, fun_evals = Newton(
        lambda x: (rosenbrock(x), rosenbrock_grad(x), rosenbrock_hessian(x)),
        x_start, alpha=1e-0, stop_tolerance=1e-10, max_steps=1e6
    )

    Xhist_ = np.array([[x[0], x[1]] for x in Xhist])

    print(
        "Found optimum at %s in %d steps (%d function evals) (true minimum is at [1,1])"
        % (Xopt, len(Xhist), fun_evals)
    )

    # Plotting
    MX, MY = np.meshgrid(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100))
    Z = np.array([MX.flatten(), MY.flatten()]).T
    VR = np.array([rosenbrock(z) for z in Z]).reshape(MX.shape)
    plt.contour(MX, MY, VR, levels=100)
    plt.plot(Xhist_[:, 0], Xhist_[:, 1], "*-k")
    plt.title("Newton-Raphson Path on Rosenbrock Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label='Rosenbrock Function Value')
    plt.show()


iris = datasets.load_iris()
print("Features: ", iris.feature_names)
print("Targets: ", iris.target_names)
petal_length = iris.data[:, iris.feature_names.index("petal length (cm)")].reshape(-1, 1)
petal_width = iris.data[:, iris.feature_names.index("petal width (cm)")].reshape(-1, 1)

# Extract the petal_length and petal_width of versicolors and virginicas

IrisX = np.hstack([np.ones_like(petal_length), petal_length, petal_width])
IrisX = IrisX[iris.target != 0, :]

# Set versicolor=0 and virginia=1
IrisY = (iris.target[iris.target != 0] - 1).reshape(-1, 1).astype(np.float64)

plt.scatter(IrisX[:, 1], IrisX[:, 2], c=IrisY, cmap="spring")
plt.xlabel("petal_length")
plt.ylabel("petal_width")

def logreg_loss(Theta, X, Y):
    #
    # Write a logistic regression cost suitable for use with fmin_l_bfgs
    #

    # reshape Theta into a column vector - lBFGS gives us a flat array
    ThetaR = Theta.reshape(X.shape[1], 1)
    predictions = sigmoid(X @ ThetaR)
    nll = -np.mean(Y * np.log(predictions + 1e-9) + (1 - Y) * np.log(1 - predictions + 1e-9))
    grad = (X.T @ (predictions - Y)) / X.shape[0]

    # reshape grad into the shape of Theta, for fmin_l_bfsgb to work
    return nll, grad.reshape(Theta.shape)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


Theta0 = np.zeros((3, ))

#
# Call a solver
#
ThetaOpt = sopt.fmin_l_bfgs_b(
    lambda Theta: logreg_loss(Theta, IrisX, IrisY), np.array(Theta0)
)[0]


#
# Now plot the found separation line
#

plt.scatter(IrisX[:, 1], IrisX[:, 2], c=IrisY.ravel(), cmap="spring")
plt.xlabel("petal_length")
plt.ylabel("petal_width")
pl_min, pl_max = plt.xlim()
pl = np.linspace(pl_min, pl_max, 1000)
plt.plot(pl, -(ThetaOpt[0] + ThetaOpt[1] * pl) / ThetaOpt[2])
plt.xlim(pl_min, pl_max)

alphas = [0, 1e-2, 1.0]
n_iter = []

def logreg_loss_with_reg(Theta, X, Y, alpha=0.):
    #
    # Copy the logistic regression code and extend it with L2 regularization
    #
    ThetaR = Theta.reshape(X.shape[1], 1)

    predictions = sigmoid(X @ ThetaR)
    nll = -np.mean(Y * np.log(predictions + 1e-9) + (1 - Y) * np.log(1 - predictions + 1e-9))

    reg_term = (alpha / 2) * np.sum(ThetaR[1:] ** 2)

    total_loss = nll + reg_term

    grad = (X.T @ (predictions - Y)) / X.shape[0]
    grad[1:] += alpha * ThetaR[1:] / X.shape[0]

    return total_loss, grad.reshape(Theta.shape)


mesh_x, mesh_y = np.meshgrid(
    np.linspace(IrisX[:, 1].min(), IrisX[:, 1].max(), 100),
    np.linspace(IrisX[:, 2].min(), IrisX[:, 2].max(), 100),
)
mesh_data = np.hstack([np.ones(mesh_x.reshape(-1, 1).shape), mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1)])

for alpha in alphas:
  Theta0 = np.zeros((3, ))

  #
  # Call a solver
  #
  ThetaOpt, _, dic = sopt.fmin_l_bfgs_b(
      lambda Theta: logreg_loss_with_reg(Theta, IrisX, IrisY, alpha=alpha), np.array(Theta0)
  )
  n_iter.append(dic['nit'])

  #
  # Now calculate probabilities for mesh_data
  #
  probs = sigmoid(mesh_data @ ThetaOpt.reshape(-1, 1))

  #
  # Plot the decision boundary
  #
  plt.contourf(mesh_x, mesh_y, probs.reshape(mesh_x.shape), cmap='spring')
  plt.scatter(IrisX[:, 1], IrisX[:, 2], c=IrisY.ravel(), cmap="spring", linewidths=1, edgecolors='black')
  plt.colorbar()
  plt.xlabel("petal_length")
  plt.ylabel("petal_width")
  pl_min, pl_max = plt.xlim()
  plt.title(fr'$\alpha$ = {alpha:.2e}')
  plt.xlim(pl_min, pl_max)
  plt.show()

  data = pd.read_csv(
      "https://gitlab.com/inzynieria_obliczeniowa_23_24/common/-/raw/main/03-house-prices-outliers.csv",
      index_col=0,
  )
  data.head()
  X = np.stack((np.ones_like(data.area), data.area)).T
  X
  Y = np.asarray(data.price)[:, None]
  Y
  # Compute least squares solution
  theta_lsq = np.linalg.inv(X.T @ X) @ (X.T @ Y)

  # Predict prices based on the model
  Y_predicted_lsq = X @ theta_lsq

  # Calculate residuals
  residuals_lsq = Y - Y_predicted_lsq

  # Plot histogram of residuals and overlay Gaussian distribution
  sns.histplot(residuals_lsq, bins=30, kde=False, color="green")
  plt.xlabel("Residuals")
  plt.ylabel("Count")

  # Fit Gaussian distribution to residuals
  mean_lsq, std_dev_lsq = norm.fit(residuals_lsq)

  # Plot the Gaussian curve on top of the histogram
  x_vals = np.linspace(residuals_lsq.min(), residuals_lsq.max(), 100)
  pdf_vals = norm.pdf(x_vals, mean_lsq, std_dev_lsq)
  plt.plot(x_vals, pdf_vals * len(residuals_lsq) * np.diff(np.histogram(residuals_lsq, bins=30)[1])[0], 'r-',
           linewidth=2)
  plt.title("Residuals with Fitted Gaussian")
  plt.show()

  # Compute the 25th and 75th quantiles of the normal distribution
  quantiles_lsq = norm.ppf([0.25, 0.75], loc=mean_lsq, scale=std_dev_lsq)
  print(f"Coefficients from Least Squares: {theta_lsq}")
  print(f"Fraction of residuals below 25th percentile: {np.mean(residuals_lsq < quantiles_lsq[0]):.6f}, "
        f"above 75th percentile: {np.mean(residuals_lsq > quantiles_lsq[1]):.6f}")
  print(f"Normal Distribution Quantiles: {quantiles_lsq}")

  # Calculate the fraction of residuals within one standard deviation
  std_confidence = np.mean(np.abs(residuals_lsq) < std_dev_lsq)
  print(f"Fraction of residuals within one standard deviation: {std_confidence:.6f}")

  from scipy.stats import norm

  # Compute least squares solution
  theta_lsq = np.linalg.inv(X.T @ X) @ (X.T @ Y)

  # Predict prices based on the model
  Y_predicted_lsq = X @ theta_lsq

  # Calculate residuals
  residuals_lsq = Y - Y_predicted_lsq

  # Plot histogram of residuals and overlay Gaussian distribution
  sns.histplot(residuals_lsq, bins=30, kde=False, color="green")
  plt.xlabel("Residuals")
  plt.ylabel("Count")

  # Fit Gaussian distribution to residuals
  mean_lsq, std_dev_lsq = norm.fit(residuals_lsq)

  # Plot the Gaussian curve on top of the histogram
  x_vals = np.linspace(residuals_lsq.min(), residuals_lsq.max(), 100)
  pdf_vals = norm.pdf(x_vals, mean_lsq, std_dev_lsq)
  plt.plot(x_vals, pdf_vals * len(residuals_lsq) * np.diff(np.histogram(residuals_lsq, bins=30)[1])[0], 'r-',
           linewidth=2)
  plt.title("Residuals with Fitted Gaussian")
  plt.show()

  # Compute the 25th and 75th quantiles of the normal distribution
  quantiles_lsq = norm.ppf([0.25, 0.75], loc=mean_lsq, scale=std_dev_lsq)
  print(f"Coefficients from Least Squares: {theta_lsq}")
  print(f"Fraction of residuals below 25th percentile: {np.mean(residuals_lsq < quantiles_lsq[0]):.6f}, "
        f"above 75th percentile: {np.mean(residuals_lsq > quantiles_lsq[1]):.6f}")
  print(f"Normal Distribution Quantiles: {quantiles_lsq}")

  # Calculate the fraction of residuals within one standard deviation
  std_confidence = np.mean(np.abs(residuals_lsq) < std_dev_lsq)
  print(f"Fraction of residuals within one standard deviation: {std_confidence:.6f}")

  # Define threshold to detect outliers (3 times the standard deviation)
  outlier_threshold = 3 * std_dev_lsq
  mask_non_outliers = np.abs(residuals_lsq) < outlier_threshold

  # Filter data points that are not outliers
  X_filtered = X[mask_non_outliers.ravel()]
  Y_filtered = Y[mask_non_outliers.ravel()]

  # Refit the model using the cleaned dataset
  theta_cleaned = np.linalg.inv(X_filtered.T @ X_filtered) @ (X_filtered.T @ Y_filtered)
  Y_pred_cleaned = X_filtered @ theta_cleaned

  # Recalculate residuals using the cleaned dataset
  residuals_cleaned = Y_filtered - Y_pred_cleaned

  # Fit a Gaussian distribution to the new residuals (after removing outliers)
  mean_cleaned, std_dev_cleaned = norm.fit(residuals_cleaned)

  # Calculate the 25th and 75th quantiles for the filtered residuals
  quantiles_cleaned = norm.ppf([0.25, 0.75], loc=mean_cleaned, scale=std_dev_cleaned)
  print(f"Theta after refitting on filtered data: {theta_cleaned}")
  print(f"Fraction of errors below 25th percentile: {np.mean(residuals_cleaned < quantiles_cleaned[0]):.6f}, "
        f"above 75th percentile: {np.mean(residuals_cleaned > quantiles_cleaned[1]):.6f}")
  print(f"Quantiles from normal distribution on filtered data: {quantiles_cleaned}")

  from scipy.optimize import fmin_l_bfgs_b


  def custom_quantile_loss(Theta, X, Y, tau=0.75):
      """Loss function for quantile regression with gradient"""
      Theta_reshaped = Theta.reshape(-1, 1)
      residuals = Y - X @ Theta_reshaped
      quantile_loss_value = np.mean((tau * np.maximum(residuals, 0)) + ((1 - tau) * np.maximum(-residuals, 0)))
      grad_value = np.mean(-X.T @ ((tau * (residuals >= 0)) + (1 - tau) * (residuals < 0)).astype(float), axis=1)
      return quantile_loss_value, grad_value


  # Use the initial parameters from the least squares solution (after filtering outliers)
  Theta_init = theta_cleaned.ravel()

  # Perform quantile regression for different quantiles (25th, 50th, 75th)
  tau_values = [0.25, 0.50, 0.75]
  Theta_results = {}

  for tau in tau_values:
      Theta_optimized, _, _ = fmin_l_bfgs_b(lambda Theta: custom_quantile_loss(Theta, X_filtered, Y_filtered, tau=tau),
                                            Theta_init)
      Theta_results[tau] = Theta_optimized
      print(f"Theta for Quantile {int(tau * 100)}: {Theta_optimized}")

      # Calculate the fraction of negative residuals for each quantile
      residuals_quantile = Y_filtered - X_filtered @ Theta_optimized.reshape(-1, 1)
      frac_neg_errors = np.mean(residuals_quantile < 0)
      print(f"Fraction of negative residuals: {frac_neg_errors:.6f} (expected {tau:.6f})")

  # Plot the data points along with regression lines for least squares and quantile regression
  plt.scatter(X_filtered[:, 1], Y_filtered, alpha=0.3, label="Data")
  plt.plot(X_filtered[:, 1], X_filtered @ theta_cleaned, label="Least Squares", color='blue')
  plt.plot(X_filtered[:, 1], X_filtered @ Theta_results[0.25], label="Quantile 25%", color='orange')
  plt.plot(X_filtered[:, 1], X_filtered @ Theta_results[0.50], label="Quantile 50%", color='green')
  plt.plot(X_filtered[:, 1], X_filtered @ Theta_results[0.75], label="Quantile 75%", color='purple')

  plt.xlabel("Area (sq meters)")
  plt.ylabel("Price")
  plt.legend()
  plt.show()

  # Estimate the 75th percentile price for a flat of 60 square meters
  flat_features = np.array([1, 60])  # [intercept term, 60 sqm area]
  price_75th_quantile = flat_features @ Theta_results[0.75]
  print(f"Estimated 75th percentile price for a 60 sqm flat: {price_75th_quantile}")

  # Assess the impact of removing outliers
  price_75th_quantile_filtered = flat_features @ Theta_results[0.75]
  print(
      f"Difference in 75th percentile price after removing outliers: {price_75th_quantile_filtered - price_75th_quantile}")