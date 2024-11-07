import torch


from .fedavgclient import FedAvgClient
from src import MetricManager


class FedavgmClient(FedAvgClient):
    def __init__(self, **kwargs):
        super(FedavgmClient, self).__init__(**kwargs)