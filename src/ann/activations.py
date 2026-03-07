import numpy as np

class ReLU:
    def forward(self, x):
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, grad):
        dx = grad.copy()
        dx[self.output <= 0] = 0
        return dx


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