import matplotlib.pyplot as plt
import numpy as np
# assignment4

# Let's define a XOR dataset

# X will be matrix of N 2-dimensional inputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1],], dtype=np.float32)
# Y is a matrix of N numners - answers
Y = np.array([[0], [1], [1], [0],], dtype=np.float32)

plt.scatter(
    X[:, 0], X[:, 1], c=Y[:, 0],
)
plt.xlabel("X[0]")
plt.ylabel("X[1]")
plt.axis("square")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SmallNet:
    def __init__(self, in_features, num_hidden, dtype=np.float32):
        self.W1 = np.zeros((num_hidden, in_features), dtype=dtype)
        self.b1 = np.zeros((num_hidden,), dtype=dtype)
        self.W2 = np.zeros((1, num_hidden), dtype=dtype)
        self.b2 = np.zeros((1,), dtype=dtype)
        self.init_params()

    def init_params(self):
        self.W1 = np.random.normal(0, 0.5, self.W1.shape)
        self.b1 = np.random.normal(0, 0.5, self.b1.shape)
        self.W2 = np.random.normal(0, 0.5, self.W2.shape)
        self.b2 = np.random.normal(0, 0.5, self.b2.shape)

    def forward(self, X, Y=None, do_backward=False):
        # TODO Problem 1: Fill in details of forward propagation

        # Input to neurons in 1st layer
        A1 = X @ self.W1.T + self.b1
        # Outputs after the sigmoid non-linearity
        O1 = sigmoid(A1)
        # Inputs to neuron in the second layer
        A2 = O1 @ self.W2.T + self.b2
        # Outputs after the sigmoid non-linearity
        O2 = sigmoid(A2)

        # When Y is none, simply return the predictions. Else compute the loss
        if Y is not None:
            loss = -Y * np.log(O2) - (1 - Y) * np.log(1.0 - O2)
            # normalize loss by batch size
            loss = loss.sum() / X.shape[0]
        else:
            loss = np.nan

        if do_backward:
            # TODO in Problem 2:
            # fill in the gradient computation
            # Please note, that there is a correspondance between
            # the forward and backward pass: with backward computations happening
            # in reverse order.
            # We save the gradients with respect to the parameters as fields of self.
            # It is not very elegant, but simplifies training code later on.

            # A2_grad is the gradient of loss with respect to A2
            # Hint: there is a concise formula for the gradient
            # of logistic sigmoid and cross-entropy loss
            A2_grad = O2 - Y
            self.b2_grad = A2_grad.sum(axis=0) / X.shape[0]
            self.W2_grad = (A2_grad.T @ O1) / X.shape[0]
            O1_grad = A2_grad @ self.W2 * O1 * (1 - O1)
            A1_grad = O1_grad * O1 * (1 - O1)
            self.b1_grad = O1_grad.sum(axis=0) / X.shape[0]
            self.W1_grad = (O1_grad.T @ X) / X.shape[0]

        return O2, loss

 # TODO Problem 1:
# Set by hand the weight values to solve the XOR problem

net = SmallNet(2, 2, dtype=np.float64)
net.W1 = np.array([[10, -10], [-10, 10]], dtype=np.float64)
net.b1 = np.array([-5, -5], dtype=np.float64)
net.W2 = np.array([[10, 10]], dtype=np.float64)
net.b2 = np.array([-5], dtype=np.float64)

# Hint: since we use the logistic sigmoid activation, the weights may need to
# be fairly large


predictions, loss = net.forward(X, Y, do_backward=True)
for x, p in zip(X, predictions):
    print(f"XORnet({x}) = {p[0]}")


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    class SmallNet:
        def __init__(self, in_features, num_hidden, dtype=np.float64):
            self.W1 = np.zeros((num_hidden, in_features), dtype=dtype)
            self.b1 = np.zeros((num_hidden,), dtype=dtype)
            self.W2 = np.zeros((1, num_hidden), dtype=dtype)
            self.b2 = np.zeros((1,), dtype=dtype)
            self.init_params()

        def init_params(self):
            # Initialize parameters to random values
            self.W1 = np.random.normal(0, 0.5, self.W1.shape)
            self.b1 = np.random.normal(0, 0.5, self.b1.shape)
            self.W2 = np.random.normal(0, 0.5, self.W2.shape)
            self.b2 = np.random.normal(0, 0.5, self.b2.shape)

        def reset_gradients(self):
            # Reset all gradients to zero
            self.W1_grad = np.zeros_like(self.W1)
            self.b1_grad = np.zeros_like(self.b1)
            self.W2_grad = np.zeros_like(self.W2)
            self.b2_grad = np.zeros_like(self.b2)

        def forward(self, X, Y=None, do_backward=False):
            # First layer computations
            A1 = np.dot(X, self.W1.T) + self.b1  # (N x num_hidden)
            O1 = sigmoid(A1)  # (N x num_hidden)

            # Second layer computations
            A2 = np.dot(O1, self.W2.T) + self.b2  # (N x 1)
            O2 = sigmoid(A2)  # (N x 1)

            if Y is not None:
                # Cross-entropy loss
                loss = -Y * np.log(O2) - (1 - Y) * np.log(1 - O2)
                loss = loss.sum() / X.shape[0]
            else:
                loss = np.nan

            if do_backward:
                # Gradient calculations for backpropagation
                A2_grad = (O2 - Y) / X.shape[0]  # Normalized gradient for batch size
                self.b2_grad = A2_grad.sum(axis=0)  # Sum over samples
                self.W2_grad = np.dot(A2_grad.T, O1)  # (1 x num_hidden)

                O1_grad = np.dot(A2_grad, self.W2)  # (N x num_hidden)
                A1_grad = O1_grad * O1 * (1 - O1)  # Sigmoid derivative (N x num_hidden)

                self.b1_grad = A1_grad.sum(axis=0)  # (num_hidden,)
                self.W1_grad = np.dot(A1_grad.T, X)  # (num_hidden x in_features)

                # Debug output for gradients
                print(f"A2_grad: {A2_grad}")
                print(f"W2_grad: {self.W2_grad}")
                print(f"b2_grad: {self.b2_grad}")
                print(f"O1_grad: {O1_grad}")
                print(f"A1_grad: {A1_grad}")
                print(f"W1_grad: {self.W1_grad}")
                print(f"b1_grad: {self.b1_grad}")

            return O2, loss


    def check_grad(net, param_name, X, Y, eps=1e-5):
        """A gradient checking routine"""

        param = getattr(net, param_name)
        param_flat_accessor = param.reshape(-1)

        grad = np.empty_like(param)
        grad_flat_accessor = grad.reshape(-1)

        # Perform forward pass to compute analytical gradients
        net.reset_gradients()  # Ensure gradients are reset
        net.forward(X, Y, do_backward=True)
        orig_grad = getattr(net, param_name + "_grad")
        assert param.shape == orig_grad.shape

        # Numerical gradient calculation
        for i in range(param_flat_accessor.shape[0]):
            orig_val = param_flat_accessor[i]
            param_flat_accessor[i] = orig_val + eps
            _, loss_positive = net.forward(X, Y)
            param_flat_accessor[i] = orig_val - eps
            _, loss_negative = net.forward(X, Y)
            param_flat_accessor[i] = orig_val
            grad_flat_accessor[i] = (loss_positive - loss_negative) / (2 * eps)

        # Print differences if the assertion fails
        if not np.allclose(grad, orig_grad, atol=1e-5):
            print(f"Gradients differ for {param_name}:")
            print(f"Numerical Gradient:\n{grad}")
            print(f"Analytical Gradient:\n{orig_grad}")
            print(f"Difference:\n{grad - orig_grad}")
            print(f"Max Difference: {np.max(np.abs(grad - orig_grad))}")

        assert np.allclose(grad, orig_grad, atol=1e-5)
        return grad, orig_grad


    # Running the gradient check
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float64)

net = SmallNet(2, 2, dtype=np.float64)

for param_name in ["W1", "b1", "W2", "b2"]:
    check_grad(net, param_name, X, Y)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SmallNet:
    def __init__(self, in_features, num_hidden, dtype=np.float64):
        self.W1 = np.zeros((num_hidden, in_features), dtype=dtype)
        self.b1 = np.zeros((num_hidden,), dtype=dtype)
        self.W2 = np.zeros((1, num_hidden), dtype=dtype)
        self.b2 = np.zeros((1,), dtype=dtype)
        self.init_params()

    def init_params(self):
        # Initialize parameters to random values
        self.W1 = np.random.normal(0, 0.5, self.W1.shape)
        self.b1 = np.random.normal(0, 0.5, self.b1.shape)
        self.W2 = np.random.normal(0, 0.5, self.W2.shape)
        self.b2 = np.random.normal(0, 0.5, self.b2.shape)

    def reset_gradients(self):
        # Reset all gradients to zero
        self.W1_grad = np.zeros_like(self.W1)
        self.b1_grad = np.zeros_like(self.b1)
        self.W2_grad = np.zeros_like(self.W2)
        self.b2_grad = np.zeros_like(self.b2)

    def forward(self, X, Y=None, do_backward=False):
        # First layer computations
        A1 = np.dot(X, self.W1.T) + self.b1
        O1 = sigmoid(A1)

        # Second layer computations
        A2 = np.dot(O1, self.W2.T) + self.b2
        O2 = sigmoid(A2)

        if Y is not None:
            # Cross-entropy loss
            loss = -Y * np.log(O2) - (1 - Y) * np.log(1 - O2)
            loss = loss.sum() / X.shape[0]
        else:
            loss = np.nan

        if do_backward:
            # Gradient calculations for backpropagation
            A2_grad = (O2 - Y) / X.shape[0]
            self.b2_grad = A2_grad.sum(axis=0)
            self.W2_grad = np.dot(A2_grad.T, O1)
            O1_grad = np.dot(A2_grad, self.W2)
            A1_grad = O1_grad * O1 * (1 - O1)
            self.b1_grad = A1_grad.sum(axis=0)
            self.W1_grad = np.dot(A1_grad.T, X)

        return O2, loss

# Define the learning rate
alpha = 0.1

# Generate data for a 3D XOR task
X3 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
               [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float64)
Y3 = np.array([[0], [1], [1], [0], [1], [0], [0], [1]], dtype=np.float64)

# Function to train the network and check success
def train_network(hidden_dim, num_trials=10):
    success_count = 0

    for _ in range(num_trials):
        # Initialize the network
        net = SmallNet(3, hidden_dim, dtype=np.float64)

        # Training loop
        for i in range(10000):
            _, loss = net.forward(X3, Y3, do_backward=True)
            for param_name in ["W1", "b1", "W2", "b2"]:
                param = getattr(net, param_name)
                param[:] -= alpha * getattr(net, param_name + "_grad")

        # Evaluate success
        predictions, _ = net.forward(X3)
        predictions = (predictions > 0.5).astype(np.float64)
        if np.array_equal(predictions, Y3):
            success_count += 1

    success_rate = success_count / num_trials
    print(f"Hidden size: {hidden_dim}, Success rate: {success_rate:.2f}")
    return success_rate

# Test different hidden sizes
for hidden_dim in [2, 3, 5, 10, 20]:
    train_network(hidden_dim)
predictions, loss = net.forward(X, Y, do_backward=True)
for x, p in zip(X, predictions):
    print(f"XORnet({x}) = {p[0]}")
X3 = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
], dtype=np.float32)

Y3 = np.array([[0], [1], [1], [0], [1], [0], [0], [1]], dtype=np.float32)

class SmallNet:
    def __init__(self, in_features, num_hidden, dtype=np.float32):
        self.W1 = np.random.randn(num_hidden, in_features).astype(dtype)  # Random initialization
        self.b1 = np.zeros((num_hidden,), dtype=dtype)  # Zero biases
        self.W2 = np.random.randn(1, num_hidden).astype(dtype)  # Random initialization
        self.b2 = np.zeros((1,), dtype=dtype)  # Zero biases

    def forward(self, X, Y=None, do_backward=False):
        A1 = np.dot(X, self.W1.T) + self.b1
        O1 = sigmoid(A1)

        A2 = np.dot(O1, self.W2.T) + self.b2
        O2 = sigmoid(A2)

        if Y is not None:
            loss = -Y * np.log(O2) - (1 - Y) * np.log(1.0 - O2)
            loss = loss.sum() / X.shape[0]
        else:
            loss = np.nan

        if do_backward:
            A2_grad = O2 - Y
            self.b2_grad = A2_grad.sum(axis=0)
            self.W2_grad = np.dot(A2_grad.T, O1)

            O1_grad = np.dot(A2_grad, self.W2)
            A1_grad = O1_grad * O1 * (1 - O1)

            self.b1_grad = A1_grad.sum(axis=0)
            self.W1_grad = np.dot(A1_grad.T, X)

        return O2, loss

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training using gradient descent
def train_3D_XOR(hidden_dim, X, Y, num_steps=100000, alpha=0.1):
    net = SmallNet(3, hidden_dim, dtype=np.float64)  # Create network with 3 inputs
    for i in range(num_steps):
        _, loss = net.forward(X, Y, do_backward=True)
        if (i % 5000) == 0:
            print(f"After {i} steps, loss = {loss}")

        # Update weights using gradient descent
        for param_name in ["W1", "b1", "W2", "b2"]:
            param = getattr(net, param_name)
            param[:] = param - alpha * getattr(net, param_name + "_grad")

    predictions, _ = net.forward(X)
    return predictions

# Example run
predictions = train_3D_XOR(10, X3, Y3)
for x, p in zip(X3, predictions):
    print(f"XORnet({x}) = {p[0]}")


    def relu(x):
        return np.maximum(0, x)


    # Define the derivative of ReLU for backpropagation
    def relu_grad(x):
        return (x > 0).astype(x.dtype)


    class ReLUNet:
        def __init__(self, in_features, num_hidden, dtype=np.float64):
            self.W1 = np.zeros((num_hidden, in_features), dtype=dtype)
            self.b1 = np.zeros((num_hidden,), dtype=dtype)
            self.W2 = np.zeros((1, num_hidden), dtype=dtype)
            self.b2 = np.zeros((1,), dtype=dtype)
            self.init_params()

        def init_params(self):
            # Initialize parameters to random values
            self.W1 = np.random.normal(0, 0.5, self.W1.shape)
            self.b1 = np.random.normal(0, 0.5, self.b1.shape)
            self.W2 = np.random.normal(0, 0.5, self.W2.shape)
            self.b2 = np.random.normal(0, 0.5, self.b2.shape)

        def reset_gradients(self):
            self.W1_grad = np.zeros_like(self.W1)
            self.b1_grad = np.zeros_like(self.b1)
            self.W2_grad = np.zeros_like(self.W2)
            self.b2_grad = np.zeros_like(self.b2)

        def forward(self, X, Y=None, do_backward=False):
            # First layer with ReLU activation
            A1 = np.dot(X, self.W1.T) + self.b1
            O1 = relu(A1)

            # Second layer with Sigmoid activation
            A2 = np.dot(O1, self.W2.T) + self.b2
            O2 = sigmoid(A2)

            if Y is not None:
                # Cross-entropy loss
                loss = -Y * np.log(O2) - (1 - Y) * np.log(1 - O2)
                loss = loss.sum() / X.shape[0]
            else:
                loss = np.nan

            if do_backward:
                # Backpropagation for output layer
                A2_grad = O2 - Y
                self.b2_grad = A2_grad.sum(0)
                self.W2_grad = np.dot(A2_grad.T, O1)

                # Backpropagation for the first layer (ReLU)
                O1_grad = np.dot(A2_grad, self.W2)
                A1_grad = O1_grad * relu_grad(A1)
                self.b1_grad = A1_grad.sum(0)
                self.W1_grad = np.dot(A1_grad.T, X)

            return O2, loss


    # Train the ReLUNet on the 3D XOR problem
    X3 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float64)
    Y3 = np.array([[0], [1], [1], [0], [1], [0], [0], [1]], dtype=np.float64)


    def train_network(hidden_dim, learning_rate=0.1, iterations=10000):
        net = ReLUNet(3, hidden_dim, dtype=np.float64)
        for i in range(iterations):
            _, loss = net.forward(X3, Y3, do_backward=True)
            if (i % 1000) == 0:
                print(f"Step {i}, Loss: {loss:.4f}")
            for param_name in ["W1", "b1", "W2", "b2"]:
                param = getattr(net, param_name)
                param[:] = param - learning_rate * getattr(net, param_name + "_grad")
        predictions, _ = net.forward(X3)
        print(f"Final Predictions: {predictions}")
        return net


    # Experiment with different hidden layer sizes to find a reliable architecture
    trained_net = train_network(hidden_dim=10)


    def generate_3d_xor():
        X = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ], dtype=np.float32)

        Y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]], dtype=np.float32)
        return X, Y


    class SmallNetWithTwoHiddenLayers:
        def __init__(self, in_features, num_hidden1, num_hidden2, dtype=np.float32):
            # Inicjalizacja wag i biasów dla pierwszej warstwy ukrytej
            self.W1 = np.random.normal(0, 0.5, (num_hidden1, in_features)).astype(dtype)
            self.b1 = np.random.normal(0, 0.5, (num_hidden1,)).astype(dtype)

            # Inicjalizacja wag i biasów dla drugiej warstwy ukrytej
            self.W2 = np.random.normal(0, 0.5, (num_hidden2, num_hidden1)).astype(dtype)
            self.b2 = np.random.normal(0, 0.5, (num_hidden2,)).astype(dtype)

            # Inicjalizacja wag i biasów dla warstwy wyjściowej
            self.W3 = np.random.normal(0, 0.5, (1, num_hidden2)).astype(dtype)
            self.b3 = np.random.normal(0, 0.5, (1,)).astype(dtype)

        def forward(self, X, Y=None, do_backward=False):
            # Pierwsza warstwa ukryta z ReLU
            A1 = X @ self.W1.T + self.b1
            O1 = relu(A1)

            # Druga warstwa ukryta z ReLU
            A2 = O1 @ self.W2.T + self.b2
            O2 = relu(A2)

            # Warstwa wyjściowa z sigmoid
            A3 = O2 @ self.W3.T + self.b3
            O3 = sigmoid(A3)

            # Obliczanie straty (cross-entropy loss)
            if Y is not None:
                loss = -Y * np.log(O3) - (1 - Y) * np.log(1 - O3)
                loss = loss.sum() / X.shape[0]
            else:
                loss = np.nan

            # Backward pass: obliczanie gradientów
            if do_backward:
                A3_grad = O3 - Y
                self.b3_grad = A3_grad.sum(axis=0) / X.shape[0]
                self.W3_grad = (A3_grad.T @ O2) / X.shape[0]

                O2_grad = A3_grad @ self.W3
                O2_grad = O2_grad * (A2 > 0)  # Pochodna ReLU dla drugiej warstwy
                self.b2_grad = O2_grad.sum(axis=0) / X.shape[0]
                self.W2_grad = (O2_grad.T @ O1) / X.shape[0]

                O1_grad = O2_grad @ self.W2
                O1_grad = O1_grad * (A1 > 0)  # Pochodna ReLU dla pierwszej warstwy
                self.b1_grad = O1_grad.sum(axis=0) / X.shape[0]
                self.W1_grad = (O1_grad.T @ X) / X.shape[0]

            return O3, loss

        # Aktualizacja parametrów gradient descent
        def update_parameters(self, alpha):
            self.W1 -= alpha * self.W1_grad
            self.b1 -= alpha * self.b1_grad
            self.W2 -= alpha * self.W2_grad
            self.b2 -= alpha * self.b2_grad
            self.W3 -= alpha * self.W3_grad
            self.b3 -= alpha * self.b3_grad


    learning_rate = 0.1
    hidden_dim1 = 10  # Rozmiar pierwszej warstwy ukrytej
    hidden_dim2 = 5  # Rozmiar drugiej warstwy ukrytej
    net = SmallNetWithTwoHiddenLayers(3, hidden_dim1, hidden_dim2, dtype=np.float64)

    X3, Y3 = generate_3d_xor()  # Generowanie danych 3D XOR

    # Trening sieci
    for i in range(100000):
        _, loss = net.forward(X3, Y3, do_backward=True)
        net.update_parameters(learning_rate)

        if i % 5000 == 0:
            print(f"after {i} steps loss={loss}")

        # Sprawdzenie sukcesu
        if loss < 0.1:
            print("Successfully trained the network!")
            break
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(x.dtype)

class VariableDepthNet:
    def __init__(self, in_features, hidden_dim, num_hidden_layers, activation='relu', dtype=np.float64):
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation

        # Initialize the activation functions
        if activation == 'relu':
            self.activation_func = relu
            self.activation_grad = relu_grad
        elif activation == 'sigmoid':
            self.activation_func = sigmoid
            self.activation_grad = sigmoid_grad
        else:
            raise ValueError("Unsupported activation function.")

        # Initialize the weights and biases for the hidden layers
        self.weights = []
        self.biases = []
        for i in range(num_hidden_layers):
            if i == 0:
                self.weights.append(np.random.normal(0, 0.5, (hidden_dim, in_features)).astype(dtype))
            else:
                self.weights.append(np.random.normal(0, 0.5, (hidden_dim, hidden_dim)).astype(dtype))
            self.biases.append(np.random.normal(0, 0.5, (hidden_dim,)).astype(dtype))

        # Initialize weights and biases for the output layer
        self.weights.append(np.random.normal(0, 0.5, (1, hidden_dim)).astype(dtype))
        self.biases.append(np.random.normal(0, 0.5, (1,)).astype(dtype))

    def forward(self, X, Y=None, do_backward=False):
        self.outputs = []
        input_to_layer = X

        # Forward pass through hidden layers
        for i in range(self.num_hidden_layers):
            A = np.dot(input_to_layer, self.weights[i].T) + self.biases[i]
            O = self.activation_func(A)
            self.outputs.append((A, O))
            input_to_layer = O

        # Forward pass through the output layer
        A_output = np.dot(input_to_layer, self.weights[-1].T) + self.biases[-1]
        O_output = sigmoid(A_output)
        self.outputs.append((A_output, O_output))

        if Y is not None:
            # Cross-entropy loss
            loss = -Y * np.log(O_output) - (1 - Y) * np.log(1 - O_output)
            loss = loss.sum() / X.shape[0]
        else:
            loss = np.nan

        if do_backward:
            # Backpropagation for the output layer
            A_grad = O_output - Y
            self.biases_grad = [A_grad.sum(0)]
            self.weights_grad = [np.dot(A_grad.T, self.outputs[-2][1])]

            # Backpropagation through hidden layers
            for i in range(self.num_hidden_layers - 1, -1, -1):
                A, O = self.outputs[i]
                if i == self.num_hidden_layers - 1:
                    O_grad = np.dot(A_grad, self.weights[i + 1])
                else:
                    O_grad = np.dot(A_grad, self.weights_grad[0])
                A_grad = O_grad * self.activation_grad(A)

                self.biases_grad.insert(0, A_grad.sum(0))
                if i == 0:
                    self.weights_grad.insert(0, np.dot(A_grad.T, X))
                else:
                    self.weights_grad.insert(0, np.dot(A_grad.T, self.outputs[i - 1][1]))

        return O_output, loss

    def update_params(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.weights_grad[i]
            self.biases[i] -= learning_rate * self.biases_grad[i]

# Training the VariableDepthNet
def train_variable_depth_net(hidden_dim, num_hidden_layers, activation='relu', learning_rate=0.1, iterations=10000):
    net = VariableDepthNet(3, hidden_dim, num_hidden_layers, activation=activation, dtype=np.float64)
    for i in range(iterations):
        _, loss = net.forward(X3, Y3, do_backward=True)
        if (i % 1000) == 0:
            print(f"Step {i}, Loss: {loss:.4f}")
        net.update_params(learning_rate)

    predictions, _ = net.forward(X3)
    print(f"Final Predictions: {predictions}")
    return net

# Experiment with different number of hidden layers and activation functions
trained_net = train_variable_depth_net(hidden_dim=10, num_hidden_layers=3, activation='relu')

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

class FeedbackAlignmentNet(VariableDepthNet):
    def __init__(self, in_features, hidden_dim, num_hidden_layers, activation='relu', dtype=np.float64):
        super().__init__(in_features, hidden_dim, num_hidden_layers, activation, dtype)

        # Initialize random fixed backward weights with correct shapes
        self.fixed_backward_weights = []
        for i in range(self.num_hidden_layers):
            self.fixed_backward_weights.append(np.random.normal(0, 0.5, (hidden_dim, hidden_dim)).astype(dtype))
        self.fixed_backward_weights.append(np.random.normal(0, 0.5, (1, hidden_dim)).astype(dtype))  # For the output layer

        # Initialize gradients for weights and biases
        self.reset_gradients()

    def reset_gradients(self):
        """Reset gradients for weights and biases to zero."""
        self.weights_grad = [np.zeros_like(w) for w in self.weights]
        self.biases_grad = [np.zeros_like(b) for b in self.biases]

    def backward(self, Y):
        # Start with the gradient at the output layer
        A_grad = self.outputs[-1][1] - Y  # Gradient for the output layer

        # Update weights and biases gradients for the output layer
        self.biases_grad[-1] = A_grad.sum(axis=0)
        self.weights_grad[-1] = np.dot(A_grad.T, self.outputs[-2][1])

        # Backpropagate through hidden layers
        for i in range(self.num_hidden_layers - 1, -1, -1):
            A, O = self.outputs[i]

            # Use fixed backward weights to propagate the gradient
            if i == self.num_hidden_layers - 1:
                O_grad = np.dot(A_grad, self.fixed_backward_weights[i + 1])  # Shape should match (batch_size, hidden_dim)
            else:
                O_grad = np.dot(A_grad, self.fixed_backward_weights[i])  # Shape should match (batch_size, hidden_dim)

            A_grad = O_grad * self.activation_grad(A)  # Element-wise multiplication

            # Update gradients for weights and biases in the current layer
            self.biases_grad[i] = A_grad.sum(axis=0)
            if i == 0:
                self.weights_grad[i] = np.dot(A_grad.T, X3)  # For the first layer, use input X3
            else:
                self.weights_grad[i] = np.dot(A_grad.T, self.outputs[i - 1][1])

    def forward(self, X, Y=None, do_backward=False):
        O_output, loss = super().forward(X, Y, do_backward=False)

        if do_backward:
            self.reset_gradients()  # Reset gradients before each backward pass
            self.backward(Y)

        return O_output, loss

# Training the Feedback Alignment Network
def train_feedback_alignment_net(hidden_dim, num_hidden_layers, activation='relu', learning_rate=0.1, iterations=10000):
    net = FeedbackAlignmentNet(3, hidden_dim, num_hidden_layers, activation=activation, dtype=np.float64)
    for i in range(iterations):
        _, loss = net.forward(X3, Y3, do_backward=True)
        if (i % 1000) == 0:
            print(f"Step {i}, Loss: {loss:.4f}")
        net.update_params(learning_rate)

    predictions, _ = net.forward(X3)
    print(f"Final Predictions: {predictions}")
    return net

# Assume X3 and Y3 are the 3D XOR inputs and labels
trained_feedback_net = train_feedback_alignment_net(hidden_dim=10, num_hidden_layers=3, activation='relu')