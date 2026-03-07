import argparse
import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import json
from ann.neural_network import build_model
from ann.objective_functions import CrossEntropyLoss, MSELoss
from ann.optimizers import SGD, RMSProp, Momentum, NAG

def load_data(dataset):
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    return X_train, y_train, X_test, y_test

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

def main():
    args = parse_arguments()

    if isinstance(args.hidden_size, int):
        args.hidden_size = [args.hidden_size] * args.num_layers
    elif len(args.hidden_size) != args.num_layers:
        args.hidden_size = [args.hidden_size[0]] * args.num_layers

    X_train, y_train, X_test, y_test = load_data(args.dataset)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    model = build_model(args)
    loss_fn = CrossEntropyLoss() if args.loss=="cross_entropy" else MSELoss()

    if args.optimizer == "sgd":
        optimizer = SGD(model.layers, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "momentum":
        optimizer = Momentum(model.layers, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "nag":
        optimizer = NAG(model.layers, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        optimizer = RMSProp(model.layers, lr=args.learning_rate, weight_decay=args.weight_decay)

    best_f1 = 0
    best_weights = None

    for epoch in range(args.epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_loss = 0
        for i in range(0, len(X_train), args.batch_size):
            X_batch = X_train_shuffled[i:i+args.batch_size]
            y_batch = y_train_shuffled[i:i+args.batch_size]

            logits = model.forward(X_batch)
            if args.loss=="cross_entropy":
                N = logits.shape[0]
                exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exps / np.sum(exps, axis=1, keepdims=True)
                batch_loss = -np.log(probs[np.arange(N), y_batch]).mean()
            else:
                y_onehot = np.zeros_like(logits)
                y_onehot[np.arange(logits.shape[0]), y_batch] = 1
                batch_loss = np.mean((logits - y_onehot)**2)
            epoch_loss += batch_loss

            grad_Ws, grad_bs = model.backward(y_batch, logits, args.loss)
            optimizer.step()

        avg_train_loss = epoch_loss / (len(X_train)//args.batch_size)

        val_logits = model.forward(X_val)
        val_preds = np.argmax(val_logits, axis=1)
        val_f1 = f1_score(y_val, val_preds, average="macro")

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = model.get_weights()
            np.save("best_model.npy", best_weights, allow_pickle=True)
            with open("best_config.json", "w") as f:
                json.dump(vars(args), f, indent=4)


    test_logits = model.forward(X_test)
    test_preds = np.argmax(test_logits, axis=1)
    test_f1 = f1_score(y_test, test_preds, average="macro")
    test_acc = np.mean(test_preds == y_test)
    print(f"Final Test F1: {test_f1:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__=="__main__":
    main()
