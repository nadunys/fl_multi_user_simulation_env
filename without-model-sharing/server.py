from collections import OrderedDict


from omegaconf import DictConfig

import torch
import time
import json

from model import Net, test


def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):

        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):

        model = Net(num_classes)

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        loss, accuracy, f1 = test(model, testloader[server_round])

        try:
            with open(f'./../results/mnist-presentation-normal/{server_round}.json', 'w') as json_file:
                round_data = { 
                    'global_loss': loss,
                    'global_accuracy': accuracy,
                    'f1_score': f1,
                    'rnd': server_round
                }
                json.dump(round_data, json_file)
        except Exception as e:
            print(e)
        
        return loss, {"accuracy": accuracy}

    return evaluate_fn

def get_on_evaluate_config(config: DictConfig):
    def evaluate_config_fn(server_round: int = 0):
        print(f'evaluate_config_fn({server_round})')
        
        return {
            "val_steps": config.val_steps,
            "batch_size": config.batch_size
        }
    
    return evaluate_config_fn