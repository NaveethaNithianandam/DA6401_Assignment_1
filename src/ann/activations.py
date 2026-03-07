import numpy as np

class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, grad_output):
        grad = grad_output.copy()
        grad[self.X <= 0] = 0
        return grad

class Sigmoid:
    def forward(self, X):
        self.out = 1 / (1 + np.exp(-X))
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out * (1 - self.out)

class Tanh:
    def forward(self, X):
        self.out = np.tanh(X)
        return self.out

    def backward(self, grad_output):
        return grad_output * (1 - self.out ** 2)
