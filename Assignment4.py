import matplotlib.pyplot as plt
import numpy as np

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
