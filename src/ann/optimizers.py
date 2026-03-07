import numpy as np

class SGD:
    def __init__(self, layers, lr=0.01, weight_decay=0.0):
        self.layers = [l for l in layers if hasattr(l, "W")]
        self.lr = lr
        self.wd = weight_decay

    def step(self):
        for layer in self.layers:
            layer.W -= self.lr * (layer.grad_W + self.wd * layer.W)
            layer.b -= self.lr * layer.grad_b

class Momentum:
    def __init__(self, layers, lr=0.01, weight_decay=0.0, momentum=0.9):
        self.layers = [l for l in layers if hasattr(l, "W")]
        self.lr = lr
        self.wd = weight_decay
        self.momentum = momentum
        self.v_W = [np.zeros_like(l.W) for l in self.layers]
        self.v_b = [np.zeros_like(l.b) for l in self.layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            self.v_W[i] = self.momentum * self.v_W[i] + self.lr * (layer.grad_W + self.wd * layer.W)
            self.v_b[i] = self.momentum * self.v_b[i] + self.lr * layer.grad_b
            layer.W -= self.v_W[i]
            layer.b -= self.v_b[i]

class NAG:
    def __init__(self, layers, lr=0.01, weight_decay=0.0, momentum=0.9):
        self.layers = [l for l in layers if hasattr(l, "W")]
        self.lr = lr
        self.wd = weight_decay
        self.momentum = momentum
        self.v_W = [np.zeros_like(l.W) for l in self.layers]
        self.v_b = [np.zeros_like(l.b) for l in self.layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            lookahead_W = layer.W - self.momentum * self.v_W[i]
            lookahead_b = layer.b - self.momentum * self.v_b[i]

            self.v_W[i] = self.momentum * self.v_W[i] + self.lr * (layer.grad_W + self.wd * lookahead_W)
            self.v_b[i] = self.momentum * self.v_b[i] + self.lr * layer.grad_b

            layer.W -= self.v_W[i]
            layer.b -= self.v_b[i]

class RMSProp:
    def __init__(self, layers, lr=0.001, weight_decay=0.0, beta=0.9, eps=1e-8):
        self.layers = [l for l in layers if hasattr(l, "W")]
        self.lr = lr
        self.wd = weight_decay
        self.beta = beta
        self.eps = eps
        self.s_W = [np.zeros_like(l.W) for l in self.layers]
        self.s_b = [np.zeros_like(l.b) for l in self.layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            self.s_W[i] = self.beta * self.s_W[i] + (1 - self.beta) * (layer.grad_W ** 2)
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * (layer.grad_b ** 2)

            layer.W -= self.lr * (layer.grad_W / (np.sqrt(self.s_W[i]) + self.eps) + self.wd * layer.W)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s_b[i]) + self.eps)
