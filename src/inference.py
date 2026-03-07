import argparse
import json
import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ann.neural_network import build_model

def load_model(model_path):
    return np.load(model_path, allow_pickle=True)

def load_data(dataset):
    if dataset == "mnist":
        (_, _), (X_test, y_test) = mnist.load_data()
    else:
        (_, _), (X_test, y_test) = fashion_mnist.load_data()
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    return X_test, y_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="mnist")
    parser.add_argument("--model_path", default="src/best_model.npy")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON file containing best model configuration")

    args = parser.parse_args()

    # Load best config if provided
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        # Override args for model building
        args.num_layers = cfg["num_layers"]
        args.hidden_size = cfg["hidden_size"]
        args.activation = cfg["activation"]
        args.weight_init = cfg["weight_init"]
    else:
        # Defaults if no config provided
        args.num_layers = 3
        args.hidden_size = [128] * args.num_layers
        args.activation = "relu"
        args.weight_init = "xavier"

    model = build_model(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    X_test, y_test = load_data(args.dataset)
    logits = model.forward(X_test)
    preds = np.argmax(logits, axis=1)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds, average="macro"))
    print("Recall:", recall_score(y_test, preds, average="macro"))
    print("F1:", f1_score(y_test, preds, average="macro"))

if __name__ == "__main__":
    main()