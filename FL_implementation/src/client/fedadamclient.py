from .fedavgclient import FedAvgClient


class FedAdamClient(FedAvgClient):
    def __init__(self, **kwargs):
        super(FedAdamClient, self).__init__(**kwargs)