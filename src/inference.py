import argparse
import numpy as np
from sklearn.metrics import f1_score
from ann.neural_network import build_model
import json

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", choices=["mnist","fashion_mnist"], default="mnist")
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--loss", "-l", choices=["cross_entropy","mse"], default="cross_entropy")
    parser.add_argument("--optimizer", "-o", choices=["sgd","momentum","nag","rmsprop"], default="rmsprop")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", "-wd", type=float, default=0.0)
    parser.add_argument("--num_layers", "-nhl", type=int, default=3)
    parser.add_argument("--hidden_size", "-sz", nargs="+", type=int, default=[128,128,128])
    parser.add_argument("--weight_init", "-wi", choices=["random","xavier","he","zeros"], default="random")
    parser.add_argument("--activation", "-a", choices=["relu","sigmoid","tanh"], default="relu")
    return parser.parse_args()


def load_data(dataset):
    from keras.datasets import mnist, fashion_mnist
    if dataset == "mnist":
        (_, _), (X_test, y_test) = mnist.load_data()
    else:
        (_, _), (X_test, y_test) = fashion_mnist.load_data()
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    return X_test, y_test


def main(args=None):
    if args is None:
        args = parse_arguments()

    X_test, y_test = load_data(args.dataset)

    with open(args.config_path, "r") as f:
        config_dict = json.load(f)

    class ArgsWrapper:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    model_args = ArgsWrapper(config_dict)

    if not hasattr(model_args, "num_layers"):
        model_args.num_layers = 1
    if not hasattr(model_args, "hidden_size"):
        model_args.hidden_size = [128] * model_args.num_layers

    if isinstance(model_args.hidden_size, int):
        model_args.hidden_size = [model_args.hidden_size] * model_args.num_layers
    elif len(model_args.hidden_size) != model_args.num_layers:
        model_args.hidden_size = [model_args.hidden_size[0]] * model_args.num_layers

    model = build_model(model_args)
    weight_dict = np.load(args.model_path, allow_pickle=True).item()
    model.set_weights(weight_dict)

    logits = model.forward(X_test)
    y_pred = np.argmax(logits, axis=1)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Test F1 Score: {test_f1:.4f}")


if __name__ == "__main__":
    main()
