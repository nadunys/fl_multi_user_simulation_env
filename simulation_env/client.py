from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl

from model import LSTM, train, test


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, vallodaer, text_field, devices, test_devices, user, user_model_path, user_model, global_model) -> None:
        super().__init__()

        self.devices = devices
        self.test_devices = test_devices
        self.user = user
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = vallodaer

        # a model that is randomly initialised at first
        self.model = LSTM(text_field)
        self.user_model = user_model
        self.global_model = global_model

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
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # a very standard looking optimiser
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # local model training
        train(self.model, optim, self.trainloader, self.valloader, epochs, len(self.trainloader)//2, f"{self.user.name}/{round}/local", self.device)

        # TODO: Should put personal and global modal fitting parts
        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        # TODO: Set global and user level evaluations
        return float(loss), len(self.valloader), {"accuracy": accuracy}


def generate_client_fn(trainloaders, valloaders, num_classes):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            vallodaer=valloaders[int(cid)],
            num_classes=num_classes,
        ).to_client()

    # return the function to spawn client
    return client_fn
