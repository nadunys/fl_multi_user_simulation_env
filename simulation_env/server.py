from omegaconf import DictConfig

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int = 0):
        print(f'fit_config_fn({server_round})')

        return {
            "lr": config.lr,
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size
        }

    return fit_config_fn


def get_on_evaluate_config(config: DictConfig):
    def evaluate_config_fn(server_round: int = 0):
        print(f'evaluate_config_fn({server_round})')
        
        return {
            "val_steps": config.val_steps,
            "batch_size": config.batch_size
        }
    
    return evaluate_config_fn

