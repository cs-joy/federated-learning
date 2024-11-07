from .fedavgclient import FedAvgClient


class FedyogiClient(FedAvgClient):
    def __init__(self, **kwargs):
        super(FedyogiClient, self).__init__(**kwargs)