import numpy as np

class SGD:
    def __init__(self, layers, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay
        self.layers = [layer for layer in layers if hasattr(layer, "W")]

    def step(self):
        for layer in self.layers:
            layer.W -= self.lr * (layer.grad_W + self.weight_decay * layer.W)
            layer.b -= self.lr * layer.grad_b

class RMSProp:
    def __init__(self, layers, lr=0.001, beta=0.9, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay

        # Only track Dense layers
        self.layers = [layer for layer in layers if hasattr(layer, "W")]

        self.cache_W = [np.zeros_like(layer.W) for layer in self.layers]
        self.cache_b = [np.zeros_like(layer.b) for layer in self.layers]

    def step(self):
        for idx, layer in enumerate(self.layers):

            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.cache_W[idx] = self.beta * self.cache_W[idx] + (1 - self.beta) * grad_W**2
            self.cache_b[idx] = self.beta * self.cache_b[idx] + (1 - self.beta) * grad_b**2

            layer.W -= self.lr * grad_W / (np.sqrt(self.cache_W[idx]) + self.eps)
            layer.b -= self.lr * grad_b / (np.sqrt(self.cache_b[idx]) + self.eps)

class Momentum:
    def __init__(self, layers, lr=0.01, beta=0.9, weight_decay=0.0):
        self.weight_decay = weight_decay
        self.layers = [layer for layer in layers if hasattr(layer, "W")]
        self.lr = lr
        self.beta = beta

        self.v_W = {}
        self.v_b = {}

    def step(self):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "W"):
                if i not in self.v_W:
                    self.v_W[i] = np.zeros_like(layer.W)
                    self.v_b[i] = np.zeros_like(layer.b)

                grad_W = layer.grad_W + self.weight_decay * layer.W
                grad_b = layer.grad_b

                self.v_W[i] = self.beta * self.v_W[i] + grad_W
                self.v_b[i] = self.beta * self.v_b[i] + grad_b

                layer.W -= self.lr * self.v_W[i]
                layer.b -= self.lr * self.v_b[i]

class NAG:
    def __init__(self, layers, lr=0.01, beta=0.9, weight_decay=0.0):
        self.layers = layers
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.layers = [layer for layer in layers if hasattr(layer, "W")]

        self.v_W = {}
        self.v_b = {}

    def step(self):
        for i, layer in enumerate(self.layers):
            if i not in self.v_W:
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)

            v_prev_W = self.v_W[i]
            v_prev_b = self.v_b[i]

            lookahead_W = layer.W - self.beta * v_prev_W
            lookahead_b = layer.b - self.beta * v_prev_b

            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.v_W[i] = self.beta * self.v_W[i] + grad_W
            self.v_b[i] = self.beta * self.v_b[i] + grad_b

            layer.W -= self.lr * self.v_W[i]
            layer.b -= self.lr * self.v_b[i]

                