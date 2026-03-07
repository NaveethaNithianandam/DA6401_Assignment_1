import numpy as np
from ann.layers import Dense
from ann.activations import ReLU
from ann.loss import CrossEntropyLoss
from ann.neural_network import NeuralNetwork

np.random.seed(42)
layers = [
    Dense(4, 5),
    ReLU(),
    Dense(5, 3)
]

model = NeuralNetwork(layers)
loss_fn = CrossEntropyLoss()

X = np.random.randn(2, 4)
y = np.array([0, 2])

logits = model.forward(X)
loss = loss_fn.forward(logits, y)

grad = loss_fn.backward()
model.backward(grad)

epsilon = 1e-5
layer = layers[0]

i, j = 0, 0  

original_value = layer.W[i, j]

layer.W[i, j] = original_value + epsilon
loss_plus = loss_fn.forward(model.forward(X), y)

layer.W[i, j] = original_value - epsilon
loss_minus = loss_fn.forward(model.forward(X), y)

layer.W[i, j] = original_value

numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
analytical_grad = layer.grad_W[i, j]

print("Numerical:", numerical_grad)
print("Analytical:", analytical_grad)
print("Close?", np.allclose(numerical_grad, analytical_grad, atol=1e-7))