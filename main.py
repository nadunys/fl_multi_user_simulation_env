import pickle
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from datatests.mnist import load_data
from client import generate_client_fn
from server import get_on_fit_config, get_on_evaluate_config
from strategy import PersonalizationStrategy


# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    ## 2. Prepare your dataset
    # data will be in the format of 
    # User {
    #     user_id: number
    #     train: Dataloader
    #     test: Dataloader
    # }
    # there will be multiple data from each user 
    dataset = load_data(cfg.num_clients, cfg.num_devices_per_client)

    ## 3. Define your clients
    client_fn = generate_client_fn(dataset)

    ## 4. Define your strategy
    avalible_clients = cfg.num_clients * cfg.num_devices_per_client
    strategy = PersonalizationStrategy(
        fraction_fit=1.0,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        fraction_evaluate=1.0,  # similar to fraction_fit, we don't need to use this argument.
        min_fit_clients=cfg.num_clients_per_round_fit,  # number of clients to sample for fit()
        min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
        min_available_clients=avalible_clients,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(cfg.config_fit), # a function to execute to obtain the configuration to send to the clients during fit()
        on_evaluate_config_fn=get_on_evaluate_config(cfg.config_evaluate),
        num_clients=avalible_clients
    )  # a function to run on the server side to evaluate the global model.

    ## 5. Start Simulation
    # With the dataset partitioned, the client function and the strategy ready, we can now launch the simulation!
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=avalible_clients,  # total number of clients
        strategy=strategy,  # our strategy of choice
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        client_resources={"num_cpus": 2, "num_gpus": 1},
        ray_init_args={"include_dashboard": True, 'num_cpus': 4, 'num_gpus': 1}
    )
    
    ## 6. Save your results
    # (This is one way of saving results, others are of course valid :) )
    # Now that the simulation is completed, we could save the results into the directory
    # that Hydra created automatically at the beginning of the experiment.
    results_path = Path(save_path) / "results.pkl"

    # add the history returned by the strategy into a standard Python dictionary
    # you can add more content if you wish (note that in the directory created by
    # Hydra, you'll already have the config used as well as the log)
    results = {"history": history, "anythingelse": "here"}

    # save the results as a python pickle
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
