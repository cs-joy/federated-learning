import os
import gc
import json
import torch
import random
import logging
import numpy as np
import concurrent.futures

from importlib import import_module
from collections import ChainMap, defaultdict

from src import init_weights, TqdmToLogger, MetricManager
from .base_server import BaseServer

logger = logging.getLogger(__name__)


class FedAvgServer(BaseServer):
    def __init__(self, args, writer, server_dataset, client_dataset, model):
        super(FedAvgServer, self).__init__()
        self.args = args
        self.writer = writer

        self.round = 0  # round indicator
        if self.args.eval_type != 'local':  # gloabl holdout set for central evaluation
            self.server_dataset = server_dataset
        self.global_model = self._init_model(model) # global model
        self.opt_kwargs = dict(lr= self.args.lr, momentum= self.args.beta1) # federation algorithm arguments
        self.curr_lr = self.args.lr # learning rate
        self.clients = self._create_clients(client_dataset)
        self.results = defaultdict(dict)  # logging results cotainer
    
    def _init_model(self, model):
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Initialize a model!')
        init_weights(model, self.args.init_type, self.args.init_gain)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully initialized the model ({self.args.model_name}; (Initialization type: {self.args.init_type.upper()}))!')
        
        return model