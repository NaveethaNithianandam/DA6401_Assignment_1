import argparse
import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from ann.layers import Dense
from ann.activations import ReLU
from ann.neural_network import NeuralNetwork, build_model
from ann.loss import CrossEntropyLoss, MSELoss
import wandb
#from ann.optimizers import SGD, RMSProp, Momentum, NAG


def load_data(dataset):
    if dataset == "mnist":
        (X, y), (X_test, y_test) = mnist.load_data()
    else:
        (X, y), (X_test, y_test) = fashion_mnist.load_data()

    X = X.reshape(X.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    return X, y, X_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-beta", "--beta", type=float, default=0.9)
    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", choices=["cross_entropy", "mse"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop"], default="rmsprop")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, default=[128])
    parser.add_argument("-a", "--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("-wi", "--weight_init", choices=["random", "xavier", "zeros"], default="xavier")
    parser.add_argument("-wp", "--wandb_project", type=str, default="da6401_assignment")

    args = parser.parse_args()

    wandb.init(project=args.wandb_project)

    config = wandb.config
    for key in config:
        setattr(args, key, config[key])
    
    if isinstance(args.hidden_size, int):
        args.hidden_size = [args.hidden_size] * args.num_layers

    X, y, X_test, y_test = load_data(args.dataset)


    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )


    model = build_model(args)
    if args.loss == "cross_entropy":
        loss_fn = CrossEntropyLoss()
    elif args.loss == "mse":
        loss_fn = MSELoss()
    
    from ann.optimizers import SGD, RMSProp, Momentum, NAG
    if args.optimizer == "sgd":
        optimizer = SGD(model.layers, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    elif args.optimizer == "momentum":
        optimizer = Momentum(model.layers, lr=args.learning_rate)
        
    elif args.optimizer == "nag":
        optimizer = NAG(model.layers, lr=args.learning_rate)
    
    elif args.optimizer == "rmsprop":
        optimizer = RMSProp(model.layers, lr=args.learning_rate)

    best_f1 = 0

    for epoch in range(args.epochs):

        epoch_loss = 0
        num_batches = 0
        grad_norm_sum = 0
        grad_count = 0
        relu_zero_count = 0
        relu_total = 0
        train_acc_sum = 0
        iteration_count = 0
        


        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for i in range(0, len(X_train), args.batch_size):

            X_batch = X_shuffled[i:i+args.batch_size]
            y_batch = y_shuffled[i:i+args.batch_size]

            logits = model.forward(X_batch)
            train_preds = np.argmax(logits, axis=1)
            train_acc_batch = np.mean(train_preds == y_batch)
            for layer in model.layers:
                if layer.__class__.__name__ == "ReLU":
                    activations = layer.output
                    relu_zero_count += np.sum(activations == 0)
                    relu_total += activations.size
            loss = loss_fn.forward(logits, y_batch)

            train_loss = loss
            train_acc_sum += train_acc_batch

            epoch_loss += train_loss
            num_batches += 1

            grad = loss_fn.backward()
    
            model.backward(grad)

            first_dense = None
            for layer in model.layers:
                if hasattr(layer, "grad_W"):
                    first_dense = layer
                    break

            if first_dense is not None:
                grad_norm = np.linalg.norm(first_dense.grad_W) 
                grad_norm_sum += grad_norm
                grad_count += 1

                if epoch == 0 and iteration_count < 50:
                    grad_dict = {}
                    for neuron_idx in range(5):
                        grad_vector = first_dense.grad_W[:, neuron_idx]  
                        grad_dict[f"grad_neuron_{neuron_idx}"] = np.linalg.norm(grad_vector)
                    wandb.log(grad_dict, step=iteration_count)
                iteration_count += 1

            optimizer.step()

        avg_train_loss = epoch_loss / num_batches
        avg_grad_norm = grad_norm_sum / max(grad_count, 1)
        dead_ratio = relu_zero_count / max(relu_total, 1)
        avg_train_acc = train_acc_sum / num_batches

        val_logits = model.forward(X_val)
        val_preds = np.argmax(val_logits, axis=1)
        f1 = f1_score(y_val, val_preds, average="macro")
        test_logits = model.forward(X_test)
        test_preds = np.argmax(test_logits, axis=1)
        test_acc = np.mean(test_preds == y_test)

        print(f"Epoch {epoch+1}, Val F1: {f1}")

        #wandb.log({
            #"epoch": epoch + 1,
            #"val_f1": f1,
            #"test_f1": test_f1
        #})

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_acc,
            "val_f1": f1,
            "test_accuracy": test_acc,
            #"test_f1": test_f1,
            "grad_norm_first_layer": avg_grad_norm,
            "relu_dead_ratio": dead_ratio
        })


        if f1 > best_f1:
            best_f1 = f1
            best_weights = model.get_weights()
            np.save("src/best_model.npy", np.array(best_weights, dtype=object), allow_pickle=True)

            import json
            best_config = vars(args)
            with open("src/best_config.json", "w") as f:
                json.dump(best_config, f, indent=4)
    
    test_logits = model.forward(X_test)
    test_preds = np.argmax(test_logits, axis=1)
    test_f1 = f1_score(y_test, test_preds, average="macro")
    test_acc = np.mean(test_preds == y_test)
    print("Final Test F1:", test_f1)
    #wandb.log({"test_accuracy": test_acc, "epoch": args.epochs})

    wandb.finish()

if __name__ == "__main__":
    main()