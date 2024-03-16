from omegaconf import DictConfig

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size
        }

    return fit_config_fn


def get_on_evaluate_config(config: DictConfig):
    def evaluate_config_fn(server_round: int):
        return {
            "val_steps": config.val_steps,
            "batch_size": config.batch_size
        }
    
    return evaluate_config_fn

