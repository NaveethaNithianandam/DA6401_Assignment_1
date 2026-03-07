import numpy as np

class CrossEntropyLoss:
    def forward(self, logits, y_true):
        self.N = logits.shape[0]
        self.y_true = y_true

        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)

        loss = -np.log(self.probs[np.arange(self.N), y_true])
        return np.mean(loss)

    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(self.N), self.y_true] -= 1
        return grad / self.N
    
class MSELoss:
    def forward(self, logits, y_true):
        self.N = logits.shape[0]
        self.num_classes = logits.shape[1]

        # convert labels to one-hot
        self.y_onehot = np.zeros((self.N, self.num_classes))
        self.y_onehot[np.arange(self.N), y_true] = 1

        self.logits = logits
        return np.mean((logits - self.y_onehot) ** 2)

    def backward(self):
        return 2 * (self.logits - self.y_onehot) / self.N