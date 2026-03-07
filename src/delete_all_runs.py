import wandb

ENTITY = "naveetha1008-"        
PROJECT = "da6401_assignment" 

api = wandb.Api()

runs = api.runs(f"{ENTITY}/{PROJECT}")

print(f"Found {len(runs)} runs. Deleting...")

for run in runs:
    print(f"Deleting: {run.name} ({run.id})")
    run.delete()

print("All runs deleted.")
