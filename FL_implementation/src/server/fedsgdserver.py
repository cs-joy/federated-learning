import logging

from .fedavgserver import FedAvgServer

logger = logging.getLogger(__name__)



class FedsgdServer(FedAvgServer):
    def __init__(self, **kwargs):
        super(FedsgdServer, self).__init__(**kwargs)