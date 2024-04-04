from typing import Dict, List, Tuple, Union, Optional
import flwr as fl
import json
import numpy as np

from flwr.server.client_proxy import ClientProxy

from init_devices import get_device_parameters
from device_selection import Device, compute_metrics, select_devices_UAC
from models.cifar import Net
from client import get_user_parameters, load_model, save_model, set_user_model_params

from flwr.common import EvaluateRes, FitRes, Parameters, FitIns, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from logging import DEBUG, INFO
from flwr.common.logger import log
from flwr.server.strategy.aggregate import aggregate

devices_list = ['laptop', 'mobile_phone', 'smart_watch']
user_list = set()
device_selection = "stat"
sample_device_per = 0.5
user_model_path = './checkpoints'
round_results_path = './results'
seq_len = 10
input_dim = 100
num_clients = 25
WAIT_TIMEOUT = 600
SEED = 42
ROUND = 5

np.random.seed(SEED)

class PersonalizationStrategy(fl.server.strategy.FedAvg):
    def __init__(
            self,
            fraction_fit: float = 0.1,
            fraction_evaluate: float = 0.1,
            min_fit_clients: int = 1,
            min_evaluate_clients: int = 1,
            min_available_clients: int = 1,
            evaluate_fn = None,
            on_fit_config_fn = None,
            on_evaluate_config_fn = None,
            initial_parameters = None,
            num_clients: int = 105,
    ) -> None:
        '''
        Initializes the personalization strategy
        '''
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters
        )
        try:
            num_device_per_user = len(devices_list)
            device_configs, device_speeds = get_device_parameters(num_clients, num_device_per_user)
            self.devices = [Device(device_configs[i], device_speeds[i // num_device_per_user - 1])
                                for i in range(num_clients)]
            self.losses = []
            self.round_times = []
            self.sampled_indices = []

            self.init_loss, self.init_drain = compute_metrics(self.devices)
            with open("{}/init.json".format('results'), 'w') as json_file:
                round_data = {
                    'rnd': 'init',
                    'total_loss': self.init_loss,
                    'dead_devices': self.init_drain,
                    'device_snapshot': [
                        {
                            'device_id': i,
                            'drain': d.drain,
                            't_local': d.t_local
                        } for i, d in enumerate(self.devices)
                    ]
                }
                json.dump(round_data, json_file)
            
            print('Initialized personalization strategy')
        except Exception as e:
            print('Something went wrong while initializing personalization strategy')
            print(e)
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        print('Start configure fit')
        if device_selection == 'none':
            print('End configure fit in device selection == none')
            return super().configure_fit(server_round, parameters, client_manager)
        else:
            try:
                sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
                success = client_manager.wait_for(num_clients=min_num_clients, timeout=WAIT_TIMEOUT)
                if not success:
                # Do not continue if not enough clients are available
                    log(INFO,
                        "Not enough clients available after timeout %s",
                        WAIT_TIMEOUT
                        )
                    return []
                
                # Sample clients
                all_clients = client_manager.all()
                cid_idx: Dict[int, str] = {}
                for idx, (cid, _) in enumerate(all_clients.items()):
                    cid_idx[idx] = cid
                    print(f"All clients cid: {cid}, idx: {idx}")

                global CURR_ROUND
                CURR_ROUND = server_round
                if server_round == 1:
                    sampled_indices = [*range(len(all_clients))]
                else:
                    num_device_per_user = len(devices_list)
                    num_sample_devices = int(len(self.devices) * sample_device_per)
                    print("NUM DEVICE PER USER SAMPLED : ", num_device_per_user)
                    sampled_indices = select_devices_UAC(num_devices=num_sample_devices, devices=self.devices, strategy=device_selection,
                                                        num_device_per_user=num_device_per_user)
                
                print('sampled indices: ', sampled_indices)
                self.sampled_indices = sampled_indices

                round_time = 0
                for idx in sampled_indices:
                    if self.devices[idx].t_local > round_time:
                        round_time = self.devices[idx].t_local

                    self.devices[idx].update_local_state(server_round)
                    self.round_times.append(round_time)

                config = {}
                if self.on_fit_config_fn is not None:
                    config = self.on_fit_config_fn(server_round)

                FitIns(parameters, config)
            except Exception as e:
                print("Something wrong with CONFIGURE FIT")
                print(e)
            print('Finish configure fit')
            return super().configure_fit(server_round, parameters, client_manager)
        
    def aggregate_fit(self,
                    server_round: int,
                    results: List[Tuple[ClientProxy, FitRes]],
                    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        try:
            print(f"Aggeregate fit is called {server_round}")
            if not results:
                return None, {}
            
            global_weights_results = []

            if device_selection != 'none':
                for client, r in results:
                    if int(client.cid) in self.sampled_indices:
                        self.devices[int(client.cid)].update_num_samples(r.num_examples)
                        self.devices[int(client.cid)].update_loss(r.metrics['loss'])
                        global_weights_results.append((parameters_to_ndarrays(r.parameters), r.num_examples))

            user_model = Net()
            devices = devices_list
            num_users = int(num_clients / len(devices))

            print(f'Devices: {devices}')
            print(f'Num users: {num_users}')

            for user in range(num_users):
                device_params = []
                for device_idx, device in enumerate(devices):
                    cid = user * len(devices) + device_idx
                    if cid in self.sampled_indices:
                        try:
                            # load user model
                            user_model_x = load_model(user_model, f'results/user/{user}')
                            user_params = get_user_parameters(user_model_x)
                            device_params.append((user_params, 10))
                            print(f'User: {user}, Device: {device}')

                        except Exception as e:
                            print('User device weights not found')
                        
                if len(device_params) > 0:
                    try:
                        user_params = aggregate(device_params)
                        user_model = Net()
                        set_user_model_params(user_model, user_params)
                        save_model(user_model, f'results/user/{user}')
                        print(f'Fit user model weights: {user}')
                    except Exception as e:
                        print(f'User model not found: {user}')
            global_weights_prime = aggregate(global_weights_results)
            return ndarrays_to_parameters(global_weights_prime), {}
                
        except Exception as e:
            print('Something went wrong with aggregrate fit')
            print(e)
            exit()

    def aggregate_evaluate(
            self,
            server_round:int, 
            results, 
            failures: List[BaseException]):
        global device_accuracies, user_accuracies
        try:
            if not results:
                super().aggregate_evaluate(server_round, results, failures)
            
            # accuracy of each client
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            f1_score = [r.metrics["f1_score"] * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]
            losses = [r.loss for _, r in results]

            # Aggregate and print custom metric
            accuracy_aggregated = sum(accuracies) / sum(examples)
            f1_score_aggregated = sum(f1_score) / sum(examples)
            loss_aggregated = np.mean(losses)
            print(f"Round {server_round} accuracy aggregated from client results: {accuracy_aggregated}")

            # User model results aggregation
            # User model performance
            # Weigh accuracy of each client by # examples used
            user_weighted_accuracies = [r.metrics["user_accuracy"] * r.num_examples for _, r in results]
            user_accuracies = [r.metrics["user_accuracy"] for _, r in results]
            user_losses = [r.metrics["user_loss"] for _, r in results]

            # Aggregate all user model test results
            user_weighted_accuracy_agg = sum(user_weighted_accuracies) / sum(examples)
            user_weighted_accuracy_var = np.var(user_weighted_accuracies)
            user_accuracy_agg = np.mean(user_accuracies)
            user_accuracy_var = np.var(user_accuracies)
            user_loss_agg = np.mean(user_losses)

            print(f"USER Round {server_round} WEIGHTED accuracy aggregated from client results: {user_weighted_accuracy_agg}")
            print(f"USER Round {server_round} accuracy aggregated from client results: {user_accuracy_agg}")

            # Device model performance
            # Weigh accuracy of each client by # examples used
            device_weighted_accuracies = [r.metrics["device_accuracy"] * r.num_examples for _, r in results]
            device_accuracies = [r.metrics["device_accuracy"] for _, r in results]
            device_losses = [r.metrics["device_loss"] for _, r in results]

            # Aggregate all user model test results
            device_weighted_accuracy_agg = sum(device_weighted_accuracies) / sum(examples)
            device_weighted_accuracy_var = np.var(device_weighted_accuracies)
            device_accuracy_agg = np.mean(device_accuracies)
            device_accuracy_var = np.var(device_accuracies)
            device_loss_agg = np.mean(device_losses)

            print(f"DEVICE Round {server_round} WEIGHTED accuracy aggregated from client results: {device_weighted_accuracy_agg}")
            print(f"DEVICE Round {server_round} accuracy aggregated from client results: {device_accuracy_agg}")

            step = server_round if server_round > 0 else ROUND + 1
            total_loss, dead_devices = compute_metrics(self.devices)

            print("MAX ROUND TIME")
            print(max(self.round_times))
            print("Sampled INDICES")
            print(self.sampled_indices)
            
            try:
                with open("{}/{}.json".format(round_results_path, server_round), 'w') as json_file:
                    round_data = {
                        'rnd': server_round,
                        'global_loss': loss_aggregated,
                        'global_accuracy': accuracy_aggregated,
                        'f1_score': f1_score_aggregated,
                        'global_accuracies': accuracies,
                        'global_examples': examples,
                        'user_loss': user_loss_agg,
                        'user_weighted_accuracy': user_weighted_accuracy_agg,
                        'user_weighted_accuracy_var': user_weighted_accuracy_var,
                        'user_accuracy': user_accuracy_agg,
                        'user_accuracy_var': user_accuracy_var,
                        'user_accuracies': user_accuracies,
                        'device_weighted_accuracy': device_weighted_accuracy_agg,
                        'device_weighted_accuracy_var': device_weighted_accuracy_var,
                        'device_accuracy': device_accuracy_agg,
                        'device_accuracy_var': device_accuracy_var,
                        'device_accuracies': device_accuracies,
                        'total_loss': total_loss,
                        'dead_devices': dead_devices,
                        'max_round_time': max(self.round_times),
                        'selected_devices': self.sampled_indices,
                        'device_snapshot': [
                            {
                                'device_id': i,
                                'stat_util': d.get_stat_utility(),
                                'device_util': d.get_device_utility(),
                                'time_util': d.get_time_utility(),
                                'overall_util': d.get_overall_utility(),
                                'drain': d.drain,
                                't_local': d.t_local
                            } for i, d in enumerate(self.devices)
                        ],
                    }

                    json.dump(round_data, json_file)
            except Exception as e:
                print(e)
                print("ROUND DATA wrong!!!")

        except Exception as e:
            print('Something went wrong in aggregate evaluate')
            print(e)
        
        return super().aggregate_evaluate(server_round, results, failures)