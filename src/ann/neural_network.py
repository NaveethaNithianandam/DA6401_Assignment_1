"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from ann.neural_layer import Dense
from ann.activations import ReLU, Sigmoid, Tanh

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, args):
        # Args: Namespace from argparse or config dict
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.weight_init = getattr(args, "weight_init", "xavier")
        self.activation = getattr(args, "activation", "relu")

        activation_map = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}
        act_class = activation_map[self.activation]

        self.layers = []
        input_dim = 784
        for h_dim in self.hidden_size:
            self.layers.append(Dense(input_dim, h_dim, self.weight_init))
            self.layers.append(act_class())
            input_dim = h_dim

        # Output layer
        self.layers.append(Dense(input_dim, 10, self.weight_init))

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true, y_pred, loss_type="cross_entropy"):
        """
        Compute gradients for all layers.
        For cross-entropy, gradient is handled here.
        """
        grad_W_list = []
        grad_b_list = []

        # Compute loss gradient
        if loss_type == "cross_entropy":
            N = y_pred.shape[0]
            exps = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
            probs = exps / np.sum(exps, axis=1, keepdims=True)
            grad = probs
            grad[np.arange(N), y_true] -= 1
            grad /= N
        elif loss_type == "mse":
            N = y_pred.shape[0]
            y_onehot = np.zeros_like(y_pred)
            y_onehot[np.arange(N), y_true] = 1
            grad = 2 * (y_pred - y_onehot) / N
        else:
            raise ValueError("Unknown loss type")

        g = grad
        for layer in reversed(self.layers):
            g = layer.backward(g)
            if hasattr(layer, "W"):
                grad_W_list.insert(0, layer.grad_W)
                grad_b_list.insert(0, layer.grad_b)

        self.grad_W = np.array(grad_W_list, dtype=object)
        self.grad_b = np.array(grad_b_list, dtype=object)

        return self.grad_W, self.grad_b

    def get_weights(self):
        d = {}
        idx = 0
        for layer in self.layers:
            if hasattr(layer, "W"):
                d[f"W{idx}"] = layer.W.copy()
                d[f"b{idx}"] = layer.b.copy()
                idx += 1
        return d

    def set_weights(self, weight_dict):
        idx = 0
        for layer in self.layers:
            if hasattr(layer, "W"):
                w_key = f"W{idx}"
                b_key = f"b{idx}"
                if w_key in weight_dict and b_key in weight_dict:
                    layer.W = np.array(weight_dict[w_key]).reshape(layer.W.shape)
                    layer.b = np.array(weight_dict[b_key]).reshape(layer.b.shape)
                else:
                    raise ValueError(f"Missing keys {w_key} or {b_key}")
                idx += 1

def build_model(args):
    return NeuralNetwork(args)

__all__ = ["NeuralNetwork", "build_model"]
 
