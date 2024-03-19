from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import torch
import flwr as fl
import copy
import os

from model import NextWordPredictor, train, test

USER_MODEL_PATH = 'results/user/'
GLOBAL_MODEL_PATH = 'results/global/model.pth'


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, vallodaer, vocab_size, devices, test_devices, user) -> None:
        super().__init__()
        print(f'Initializing flower client {user}')

        self.devices = devices
        self.test_devices = test_devices
        self.user = user
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = vallodaer

        # a model that is randomly initialised at first and load user and global model
        self.model = NextWordPredictor(vocab_size, 10, 10)

        user_model_path = f'{USER_MODEL_PATH}{user}/model.pth'
        if os.path.exists(user_model_path):
            print(f'Loading user model {user_model_path}')
            new_model = copy.deepcopy(self)
            new_model.load_state_dict(torch.load(user_model_path))
            self.user_model = new_model
            print(f'User model loaded {user_model_path}')

        if os.path.exists(GLOBAL_MODEL_PATH):
            print(f'Loading global model {user}')
            new_model = copy.deepcopy(self)
            new_model.load_state_dict(torch.load(user_model_path))
            self.global_model = new_model
            print(f'Global model loaded {user}')
        
        print(f'Initialized flower client {user}')

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        lr = config["lr"]
        epochs = config["local_epochs"]

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        criterion = CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=lr)

        # local model training
        loss = train(self.model, self.trainloader, criterion, optimizer, epochs)

        # TODO: Should put personal and global modal fitting parts
        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)
        return self.get_parameters({}), len(self.trainloader), {'loss': loss}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        criterion = CrossEntropyLoss()
        loss = test(self.model, self.valloader, self.device, criterion)
        # TODO: Set global and user level evaluations
        return float(loss), len(self.valloader)


def generate_client_fn(train):
    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())
       
        print({"generate_client_fn/client_fn cid": cid})
        return FlowerClient(
            trainloader=train[int(cid)][0],
            vallodaer=train[int(cid)][1],
            vocab_size=train[int(cid)][3],
            devices=[],
            test_devices=[],
            user=train[int(cid)][2],
        ).to_client()

    # return the function to spawn client
    return client_fn
