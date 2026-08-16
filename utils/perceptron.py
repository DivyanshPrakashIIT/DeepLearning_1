import numpy as np

def sigmoid(net):
    net = np.clip(net, -500, 500)  # overflow se bachne ke liye
    return 1.0 / (1.0 + np.exp(-net))

def sigmoid_derivative(y):
    return y * (1 - y)

def tanh(net):
    return np.tanh(net)

def tanh_derivative(y):
    return 1 - y**2

def linear(net):
    return net

def linear_derivative(y):
    return np.ones_like(y)


class Perceptron:
    def __init__(self, activation='sigmoid', lr=0.01, epochs=1000, seed=1):
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.activation_name = activation

        if activation == 'sigmoid':
            self.act_fn = sigmoid
            self.act_deriv = sigmoid_derivative
        elif activation == 'tanh':
            self.act_fn = tanh
            self.act_deriv = tanh_derivative
        elif activation == 'linear':
            self.act_fn = linear
            self.act_deriv = linear_derivative

        self.w = None
        self.b = None
        self.error_history = []

    def fit(self, X, t):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.seed)
        self.w = rng.uniform(-0.5, 0.5, n_features)
        self.b = rng.uniform(-0.5, 0.5)

        for epoch in range(self.epochs):
            net = X @ self.w + self.b
            y = self.act_fn(net)
            error = t - y

            deriv = self.act_deriv(y)
            grad_common = -2 * error * deriv
            grad_w = (X.T @ grad_common) / n_samples
            grad_b = np.sum(grad_common) / n_samples

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            avg_error = np.mean(error ** 2)
            self.error_history.append(avg_error)

        return self

    def predict_raw(self, X):
        net = X @ self.w + self.b
        return self.act_fn(net)