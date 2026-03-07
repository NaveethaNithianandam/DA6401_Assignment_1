import wandb

sweep_config = {
    'method': 'random', 
    'name': 'mlp_hyperparam_sweep',
    'metric': {'name': 'val_f1', 'goal': 'maximize'},
    'parameters': {
        'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop']},
        'learning_rate': {'values': [0.001, 0.005, 0.01, 0.05]},
        'num_layers': {'values': [1, 2, 3]},
        'hidden_size': {'values': [64, 128]},
        'activation': {'values': ['relu', 'tanh', 'sigmoid']},
        'weight_init': {'values': ['random', 'xavier']},
        'batch_size': {'values': [32, 64]},
        'epochs': {'value': 20}  
    }
}

sweep_id = wandb.sweep(sweep_config, project='da6401_assignment')