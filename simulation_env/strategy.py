from typing import Dict, List, Tuple, Union, Optional
import flwr as fl
import json
import numpy as np

from flwr.server.client_proxy import ClientProxy

from init_devices import get_device_parameters
from device_selection import Device, compute_metrics, select_devices_UAC
from model import NextWordPredictor

from flwr.common import EvaluateRes, FitRes, Parameters, FitIns, Scalar, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from logging import DEBUG, INFO
from flwr.common.logger import log

devices_list = ['laptop', 'mobile_phone', 'smart_watch']
user_list = set()
device_selection = "overall"
sample_device_per = 0.5
user_model_path = './checkpoints'
seq_len = 10
input_dim = 100
WAIT_TIMEOUT = 600
SEED = 42

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
        if device_selection == 'none':
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
                msg = "Round %s, sample %s clients (based on device selection criteria)"
                log(DEBUG, msg, str(server_round), str(sample_size))
                all_clients = client_manager.all()
                cid_idx: Dict[int, str] = {}
                for idx, (cid, _) in enumerate(all_clients.items()):
                    cid_idx[idx] = cid
                    # print("All clients cid: {}, idx: {}".format(cid, idx))

                global CURR_ROUND
                CURR_ROUND = server_round
                if server_round == 1:
                    # sampled_indices = select_devices(rnd, devices=self.devices, strategy='random')
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
            return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(self,
                    server_round: int,
                    results: List[Tuple[ClientProxy, FitRes]],
                    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        try:
            print("Aggeregate fit is called")
            if not results:
                return None, {}
            
            global_weights_results = {}

            if device_selection != 'none':
                for client, r in results:
                    if int(client.cid) in self.sampled_indices:
                        self.devices[int(client.cid)].update_num_samples(r.num_examples)
                        self.devices[int(client.cid)].update_loss(r.metrics['loss'])
                        global_weights_results.append((parameters_to_ndarrays(r.parameters), r.num_examples))

                    if user_model_path != 'no_personal' and user_model_path != 'local_finetuning':
                        input_shape = (seq_len, input_dim)
                        user_model = NextWordPredictor()

                    devices = devices_list
                    params_by_user = {u: [] for u in list(user_list)}
                    # TODO: load user, device and set weights

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
                return None
            
            # accuracy of each client
            accuracies = []
            examples = []
            losses = []

            # global model accurary and loss
            aggregated_accuracy = sum(accuracies) / sum(examples)
            aggergated_los = np.mean(losses)

            print(f'For round: {server_round} aggregated accuracy is: {aggregated_accuracy}')

            if user_model_path != 'no_personal' and user_model_path != 'local_finetuning':
                # create user and device weighted accuracy and loss
                print('create user and device weighted accuracies')
            
            if user_model_path != 'no_personal' and user_model_path != 'local_finetuning':
                # write round results with user results
                print('write round aand user results')
            else:
                # write only round results
                print('write only round results')

        except Exception as e:
            print('Something went wrong in aggregate evaluate')
            print(e)
        
        return super().aggregate_evaluate(server_round, results, failures)