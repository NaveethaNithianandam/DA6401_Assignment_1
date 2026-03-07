class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X  # return logits

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_weights(self):
        weights = []
        for layer in self.layers:
            if hasattr(layer, "W"):
                weights.append((layer.W, layer.b))
        return weights

    def set_weights(self, weights):
        idx = 0
        for layer in self.layers:
            if hasattr(layer, "W"):
                layer.W, layer.b = weights[idx]
                idx += 1


from ann.layers import Dense
from ann.activations import ReLU, Sigmoid, Tanh

def build_model(args):
    layers = []

    input_dim = 784
    hidden_dims = args.hidden_size

    if len(hidden_dims) != args.num_layers:
        raise ValueError("Number of hidden layers must match hidden_size list length")

    activation_map = {
        "relu": ReLU,
        "sigmoid": Sigmoid,
        "tanh": Tanh
    }

    act_class = activation_map[args.activation]

    for hidden_dim in hidden_dims:
        layers.append(Dense(input_dim, hidden_dim, args.weight_init))
        layers.append(act_class())
        input_dim = hidden_dim

    layers.append(Dense(input_dim, 10, args.weight_init))

    return NeuralNetwork(layers)