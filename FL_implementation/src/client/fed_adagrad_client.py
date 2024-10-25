from .fed_avg_client import FedAvgClient


class FedAdagradClient(FedAvgClient):
    def __init__(self, **kwargs):
        super(FedAdagradClient, self).__init__(**kwargs)