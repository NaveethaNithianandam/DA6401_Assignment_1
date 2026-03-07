import numpy as np

class Dense:
    def __init__(self, in_dim, out_dim, weight_init="xavier"):
        if weight_init == "he":
            self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
        elif weight_init == "xavier":
            limit = np.sqrt(6 / (in_dim + out_dim))
            self.W = np.random.uniform(-limit, limit, (in_dim, out_dim))
        elif weight_init == "zeros":
            self.W = np.zeros((in_dim, out_dim))
        else:  
            self.W = np.random.randn(in_dim, out_dim) * 0.01

        self.b = np.zeros(out_dim)
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, grad_output):
        self.grad_W = self.X.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0)
        return grad_output @ self.W.T