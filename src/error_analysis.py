

import numpy as np
import pandas as pd
import json
from keras.datasets import mnist, fashion_mnist
from ann.neural_network import build_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
import types


with open("src/best_config.json", "r") as f:
    config = json.load(f)

import argparse
config = argparse.Namespace(**config)

if config.dataset == "mnist":
    (_, _), (X_test, y_test) = mnist.load_data()
else:
    (_, _), (X_test, y_test) = fashion_mnist.load_data()

X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

model = build_model(config)
best_weights = np.load("src/best_model.npy", allow_pickle=True)
model.set_weights(best_weights)

logits = model.forward(X_test)
y_pred = np.argmax(logits, axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Best Model")
plt.show()

wrong_idx = np.where(y_test != y_pred)[0]
sample_idx = random.sample(list(wrong_idx), 16) 

plt.figure(figsize=(10,10))
for i, idx in enumerate(sample_idx):
    plt.subplot(4,4,i+1)
    plt.imshow(X_test[idx].reshape(28,28), cmap='gray')
    plt.title(f"T:{y_test[idx]} P:{y_pred[idx]}")
    plt.axis('off')
plt.suptitle("Sample Misclassifications")
plt.show()


cm_df = pd.DataFrame(cm)
cm_df = cm_df.div(cm_df.sum(axis=1), axis=0)  
cm_df.values[[np.arange(len(cm))]*2] = 0

top_confusions = np.unravel_index(np.argsort(cm_df.values.ravel())[-3:], cm_df.shape)
for i,j in zip(*top_confusions):
    print(f"Most confused: True {i} predicted as {j} ({cm_df.values[i,j]:.2f})")