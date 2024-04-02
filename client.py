from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl
import copy
import os

from models.cifar import Net, train, test

USER_MODEL_PATH = 'results/user/'
GLOBAL_MODEL_PATH = 'results/global'


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, devices, test_devices, user_id, cid) -> None:
        super().__init__()
        print(f'Initializing flower client {user_id}')

        self.devices = devices
        self.test_devices = test_devices
        self.user = user_id
        self.cid = cid
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = valloader

        # a model that is randomly initialised at first and load user and global model
        self.model = Net()

        self.user_model_path = f'{USER_MODEL_PATH}{user_id}'
        self.global_model_path = f'{GLOBAL_MODEL_PATH}'

        user_model = load_model(Net(), self.user_model_path)
        if user_model is not None:
            self.model = copy.deepcopy(user_model)
        
        print(f'Initialized flower client {user_id}')

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        print(f'Fitting {self.user} {self.device} {self.user_model_path}')
        lr = config["lr"]
        epochs = config["local_epochs"]

        user_model = load_model(Net(), self.user_model_path)
        global_model = load_model(Net(), self.global_model_path)
        print(f'User and gloabal models loaded: {self.user}')

        if user_model is None:
            user_model = copy.deepcopy(self.model)
            print(f'Current model got copied for user_model {self.user}')
        
        if global_model is None:
            global_model = copy.deepcopy(user_model)
            print(f'Current model got copied for global_model {self.user}')

        # copy parameters sent by the server into client's local model
        print(f'Starting to set params {self.user}')
        self.set_parameters(parameters)
        print(f'Self params were modified {self.user}')
        set_user_model_params(user_model, parameters)
        print(f'User params were modified {self.user}')

        # local model training
        print(f'Local training started for user {self.user}')
        loss = train(self.model, self.trainloader, epochs, lr)
        print(f'Training done for user {self.user} ')
        set_user_model_params(user_model, self.get_parameters({}))
        print(f'Local params were set to user model {self.user}')

        # save models
        save_model(user_model, self.user_model_path)
        save_model(global_model, self.global_model_path)

        print(f'Finish client fitting {self.user} {self.device} {self.user_model_path}')
        return self.get_parameters({}), len(self.trainloader), {'loss': loss}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        try:
            print(f'starting client evaluate {config}')
            self.set_parameters(parameters)
            print('self params set')
            loss, accuracy, f1_score = test(self.model, self.valloader)
            print(f'loss: {loss} accuracy: {accuracy} after calling test')
            user_loss, user_accuracy, user_f1_score = loss, accuracy, f1_score
            device_loss, device_accuracy, device_f1_score = loss, accuracy, f1_score

            user_model = load_model(Net(), self.user_model_path)
            if user_model is not None:
                set_user_model_params(user_model, parameters)
                user_loss, user_accuracy, user_f1_score = test(user_model, self.valloader)
            return loss, len(self.valloader), {"accuracy": accuracy, "user": self.user,
                                            "user_accuracy": user_accuracy, "user_loss": user_loss,
                                            "device_accuracy": device_accuracy, "device_loss": device_loss,
                                            "f1_score": f1_score, "user_f1_score": user_f1_score, "device_f1_score": device_f1_score}
        except Exception as e:
            print('Something wrong with client evaluation')
            print(e)        


def generate_client_fn(dataset):
    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())
       
        print({"generate_client_fn/client_fn cid": cid})
        return FlowerClient(
            trainloader=dataset[int(cid)]['train'],
            valloader=dataset[int(cid)]['test'],
            devices=[],
            test_devices=[],
            user_id=dataset[int(cid)]['user_id'],
            cid=cid
        ).to_client()

    # return the function to spawn client
    return client_fn

def load_model(model: Net, model_path: str):
    if os.path.exists(model_path):
        print(f'Loading model {model_path}')
        new_model = copy.deepcopy(model)
        new_model.load_state_dict(torch.load(f'{model_path}/model.pth'))
        
        print(f'Model loaded {model_path}')
        return new_model
    
    return None

def save_model(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), f'{path}/model.pth')

def set_user_model_params(model: Net, params):
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def get_user_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
    