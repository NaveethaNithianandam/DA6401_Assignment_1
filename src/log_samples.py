import wandb
import torch
from torchvision import datasets, transforms

wandb.init(project="da6401_assignment", name="mnist_data_exploration")

transform = transforms.ToTensor()
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

samples = {i: [] for i in range(10)}

for img, label in dataset:
    if len(samples[label]) < 5:
        samples[label].append(img)
    if all(len(samples[i]) == 5 for i in range(10)):
        break

table = wandb.Table(columns=["Digit", "Image"])

for digit in range(10):
    for img in samples[digit]:
        table.add_data(str(digit), wandb.Image(img))

print("Logging MNIST table to W&B...")
wandb.log({"MNIST Sample Images": table})
print("Logged successfully!")

wandb.finish()