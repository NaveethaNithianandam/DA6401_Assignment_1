import numpy as np
from keras.datasets import mnist, fashion_mnist
import wandb

DATASET = "mnist"  
SAMPLES_PER_CLASS = 5
PROJECT_NAME = "da6401_assignment"
RUN_NAME = "data_exploration"

if DATASET == "mnist":
    (X_train, y_train), _ = mnist.load_data()
elif DATASET == "fashion":
    (X_train, y_train), _ = fashion_mnist.load_data()
else:
    raise ValueError("Dataset must be 'mnist' or 'fashion'")

def sample_images_per_class(X, y, samples_per_class=5):
    selected_images = []
    selected_labels = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        chosen = np.random.choice(idx, samples_per_class, replace=False)
        selected_images.extend(X[chosen])
        selected_labels.extend(y[chosen])
    return np.array(selected_images), np.array(selected_labels)

sampled_images, sampled_labels = sample_images_per_class(X_train, y_train, SAMPLES_PER_CLASS)

wandb.init(project=PROJECT_NAME, name=RUN_NAME)


table = wandb.Table(columns=["image", "label"])
for img, lbl in zip(sampled_images, sampled_labels):
    table.add_data(wandb.Image(img), int(lbl))

wandb.log({"Sample Images per Class": table})
print(f"Logged {len(sampled_images)} images to W&B. Check your project: {PROJECT_NAME}")

wandb.finish()