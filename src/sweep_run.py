import wandb
from train import main

sweep_id = "vx0lowuh"

wandb.agent(
    sweep_id,
    function=main,
    entity="naveetha1008-",
    project="da6401_assignment"
)