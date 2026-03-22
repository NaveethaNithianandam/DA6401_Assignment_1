### DA6401 Assignment 1
## NumPy Implementation of a Multi-Layer Perceptron (MLP)

This project implements a configurable Multi-Layer Perceptron (MLP) from scratch using NumPy only for classification of the MNIST and Fashion-MNIST datasets.

The implementation includes:
* Forward propagation
* Backpropagation
* Multiple optimizers
* Configurable architectures
* Weight initialization methods
* Gradient logging
* Hyperparameter sweeps
* Experiment tracking using Weights & Biases

No deep learning frameworks such as PyTorch, TensorFlow, or JAX were used.

## Project Structure

```
da6401_assignment_1
│
├── src
│   ├── ann
│   │   ├── activations.py
│   │   ├── neural_layer.py
│   │   ├── objective_function.py
│   │   ├── neural_network.py
│   │   └── optimizers.py
│   │
│   ├── train.py
│   ├── inference.py
│   ├── gradient_check.py
│   ├── data_exploration.py
│   ├── error_analysis.py
│   ├── sweep_config.py
│   ├── sweep_run.py
│   ├── delete_all_runs.py
│   ├── FMNIST.py
│   ├── log_samples.py
│   │
│   ├── best_model.npy
│   └── best_config.json
│
├── README.md
├── requirements.txt
├── .gitignore
├── Figure_2_8.png
└── Figure_2_8i.png
```


**Installation**

Install required dependencies: pip install -r requirements.txt
Required libraries:

* numpy
* scikit-learn
* matplotlib
* keras
* wandb

**Training**

Run training with default configuration: python src/train.py --dataset mnist

Example custom configuration:

python src/train.py \
-d mnist \
-e 10 \
-b 64 \
-l cross_entropy \
-o rmsprop \
-lr 0.001 \
-nhl 3 \
-sz 128 \
-a relu \
-wi random



**Inference**

Run inference using the saved model:

python src/inference.py \
--dataset mnist \
--model_path src/best_model.npy \
--config src/best_config.json

This prints:

* Accuracy
* Precision
* Recall
* F1-score

# Weights and Biases Report

W&B dashboard containing all experiments:

**Main Project Workspace**
`https://wandb.ai/naveetha1008-/da6401_assignment`

**Fashion-MNIST Transfer Experiments**
`https://wandb.ai/naveetha1008-/fashion_mnist_transfer`

## 2.1 Data Exploration and Class Distribution

*W&B Visualization:*

`https://wandb.ai/naveetha1008-/da6401_assignment/panel/pokex37lx`

Five samples from each digit class were visualized using a W&B table.

Visually Similar Digits

Upon visual inspection:

* Digits 3 and 5 appear visually similar in some handwriting styles.
* Digits 4 and 9 may look similar in some handwriting styles.
* Digits 7 and 1 can sometimes be confused when written without a cross stroke.
* Digits 0 and 6 may resemble each other sometimes.

**Impact on Model Performance**

1. Visual similarity can impact classification performance by:
2. Increasing misclassification between similar digits.
3. Reducing per-class accuracy.
4. Causing feature overlap in the learned representations.

Requiring the model to learn fine-grained stroke-level features.

## 2.2 Hyperparameter Sweep

**Sweep dashboard:**`https://wandb.ai/naveetha1008-/da6401_assignment/sweeps/vx0lowuh`

**Configuration:**

Parameter	Value
Activation	ReLU
Batch Size	64
Epochs	20
Hidden Size	128
Learning Rate	0.05
Number of Layers	1
Optimizer	NAG
Weight Initialization	Random

**Key Observation**

The learning rate had the most significant impact on validation accuracy.

In the parallel coordinates plot, high validation F1 scores were concentrated around learning rates of approximately 0.005–0.01, while larger values such as 0.05 resulted in poor performance.

This indicates that the training process is highly sensitive to the learning rate.

## 2.3 Optimizer Showdown

**Workspace panel:**`https://wandb.ai/naveetha1008-/da6401_assignment/workspace/panel/bu5a1pvka`

Four optimizers were compared:
* SGD
* Momentum
* NAG
* RMSProp

All experiments used:

3 hidden layers
128 neurons each
ReLU activation

Among the four optimizers, RMSProp minimized the loss the fastest during the first five epochs. Because adapts the learning rate for each parameter individually by maintaining a running average of squared gradients. This helps to:

1. Reduce oscillations during optimization.
2. Handle varying gradient magnitudes across parameters.
3. Improve stability during training.

Since image classification problems often contain gradients with varying magnitudes, adaptive optimizers like RMSProp often converge faster than standard SGD.

## 2.4 Vanishing Gradient Analysis

Experiments were conducted using RMSProp with two activation functions:
1. Sigmoid
2. ReLU

Gradient norms of the first hidden layer were logged during training.

* Sigmoid activation exhibited vanishing gradients, where gradient norms decreased significantly as training progressed.
* ReLU maintained stronger gradients, allowing better gradient flow during backpropagation.

Sigmoid activations can lead to vanishing gradient problems in deeper networks, while ReLU mitigates this issue due to its linear gradient for positive inputs.

## 2.5 Dead Neuron Investigation

**Workspace panel:** `https://wandb.ai/naveetha1008-/da6401_assignment/workspace?panelDisplayName=relu_dead_ratio`

Experiments were conducted using:

ReLU activation
Learning rate = 0.1
SGD optimizer
Observation

* The proportion of zero activations increased steadily during training, reaching approximately 47% by the final epoch. This indicates that many neurons became inactive (dead neurons).This is because, ReLU outputs zero for negative inputs and its derivative is also zero in that region. Once a neuron becomes inactive, it stops receiving gradient updates.
* When the same experiment was conducted with Tanh activation, the proportion of zero activations remained negligible. Tanh allows gradients to flow even for negative inputs, resulting in more stable training.

## 2.6 Loss Function Comparison

**Workspace panel:**`https://wandb.ai/naveetha1008-/da6401_assignment/workspace/panel/4i232ghvj`

Two models were trained with identical architectures using:

1. Mean Squared Error (MSE)
2. Cross-Entropy Loss

Cross-Entropy converged significantly faster than MSE. This because, it directly measures the difference between the predicted probability distribution and the true distribution. This produces larger and more informative gradients, enabling faster learning. MSE treats classification as a regression task and often produces smaller gradients, slowing convergence.

## 2.7 Global Performance Analysis

**Visualization:** `https://wandb.ai/naveetha1008-/da6401_assignment/workspace/panel/7krg3mzap`

The overlay plot shows runs where:

Training Accuracy → High
Test Accuracy → Lower

This indicates overfitting, where the model memorizes training data but fails to generalize to unseen samples. This typically occurs due to:
* Excessive model capacity
* Insufficient regularization

## 2.8 Error Analysis

* A confusion matrix was generated for the best performing model. (da6401_assignment_1/Figure_2_8.png)
* Misclassified images were visualized to understand model mistakes. (da6401_assignment_1/Figure_2_8i.png)

## 2.9 Weight Initialization and Symmetry

**Workspace panel:**`https://wandb.ai/naveetha1008-/da6401_assignment/workspace/panel/xqar6k28r`

Two initialization strategies were compared:
1. Zero Initialization
2. Xavier Initialization


* With zero initialization, gradients for all neurons were identical and overlapped perfectly in the plot.
* This occurs because every neuron starts with the same weights and receives identical updates during training.
* As a result, all neurons learn the same features.
* This phenomenon is known as symmetry.
* With Xavier initialization, neurons start with different weights and produce different gradients.
* This breaks symmetry and allows neurons to learn different features.

Thus, symmetry breaking through proper weight initialization is essential for training multilayer perceptrons.

## 2.10 Fashion-MNIST Transfer Challenge

**Experiments:** `https://wandb.ai/naveetha1008-/fashion_mnist_transfer`

Three configurations were selected based on MNIST experiments.

Run	Architecture	Activation	Optimizer	Weight Init	Test Accuracy
azure_best	1 × 128	ReLU	NAG	Random	0.8734
resilient_second	2 × 128	ReLU	NAG	Xavier	0.8764
treasured_third	3 × 128	Tanh	NAG	Xavier	0.8662
Observation

* The configuration that worked best for MNIST did not achieve the best performance on Fashion-MNIST.
* The 2-layer architecture with ReLU and Xavier initialization achieved the highest accuracy.
* Fashion-MNIST contains more complex visual patterns compared to handwritten digits.
* Deeper architectures allow the network to learn hierarchical features, improving performance on more complex datasets.


**The best model was saved as:**`src/best_model.npy`

**Configuration:**

1 hidden layer
128 neurons
ReLU activation
NAG optimizer
learning rate = 0.01

**Final Testing**

Training test:

python src/train.py --dataset mnist

Inference test:

python src/inference.py \
--dataset mnist \
--model_path src/best_model.npy \
--config src/best_config.json


**W&B Report :** `https://wandb.ai/naveetha1008-/da6401_assignment/reports/DA6401-Assignment-1-MLP-from-Scratch--VmlldzoxNjEzMTg4NA?accessToken=z54noyyuma3ifd6brm33wutp66ktzg4vm15q13q9hbgzrsvu5v3j8aii4q5bfufm`

**Github link :** `https://github.com/NaveethaNithianandam/DA6401_Assignment_1`
