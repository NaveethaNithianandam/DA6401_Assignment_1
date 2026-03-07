import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ann.neural_network import build_model
from ann.loss import CrossEntropyLoss
from ann.optimizers import SGD, RMSProp, Momentum, NAG
import wandb

class ConfigObj:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

(X, y), (X_test, y_test) = fashion_mnist.load_data()
X = X.reshape(X.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

configs = [
    {
        "name": "azure_best",
        "num_layers": 1,
        "hidden_size": [128],
        "optimizer": "nag",
        "activation": "relu",
        "lr": 0.05,
        "batch_size": 64,
        "epochs": 20,
        "weight_init": "random"
    },
    {
        "name": "resilient_second",
        "num_layers": 2,
        "hidden_size": [128, 128],
        "optimizer": "nag",
        "activation": "relu",
        "lr": 0.05,
        "batch_size": 64,
        "epochs": 20,
        "weight_init": "xavier"
    },
    {
        "name": "treasured_third",
        "num_layers": 3,
        "hidden_size": [128, 128, 128],
        "optimizer": "nag",
        "activation": "tanh",
        "lr": 0.05,
        "batch_size": 64,
        "epochs": 20,
        "weight_init": "xavier"
    }
]

results = []

for cfg in configs:
    wandb.init(project="fashion_mnist_transfer", name=cfg["name"])
    cfg_obj = ConfigObj(cfg)

    model = build_model(cfg_obj)
    loss_fn = CrossEntropyLoss()

    opt_name = cfg_obj.optimizer.lower()
    if opt_name == "sgd":
        optimizer = SGD(model.layers, lr=cfg_obj.lr)
    elif opt_name == "rmsprop":
        optimizer = RMSProp(model.layers, lr=cfg_obj.lr)
    elif opt_name == "momentum":
        optimizer = Momentum(model.layers, lr=cfg_obj.lr)
    elif opt_name == "nag":
        optimizer = NAG(model.layers, lr=cfg_obj.lr)

    for epoch in range(cfg_obj.epochs):
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for i in range(0, len(X_train), cfg_obj.batch_size):
            X_batch = X_shuffled[i:i+cfg_obj.batch_size]
            y_batch = y_shuffled[i:i+cfg_obj.batch_size]

            logits = model.forward(X_batch)
            loss = loss_fn.forward(logits, y_batch)

            grad = loss_fn.backward()
            model.backward(grad)
            optimizer.step()

        val_logits = model.forward(X_val)
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"Run {cfg['name']}, Epoch {epoch+1}, Val Accuracy: {val_acc:.4f}")
        wandb.log({"epoch": epoch + 1, "val_accuracy": val_acc})

    test_logits = model.forward(X_test)
    test_preds = np.argmax(test_logits, axis=1)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"{cfg['name']} Test Accuracy: {test_acc:.4f}")
    results.append({"config": cfg["name"], "test_acc": test_acc})
    wandb.log({"test_accuracy": test_acc})
    wandb.finish()

print("All configurations results:")
for r in results:
    print(r)