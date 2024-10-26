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
    
    def _get_algorithm(self, model, **kwargs):
        ALGORITHM_CLASS = import_module(f'..algorithm.{self.args.algorithm}', package= __package__).__dict__[f'{self.args.algorithm.title()}Optimizer']
        optimizer = ALGORITHM_CLASS(params= model.parameters(), **kwargs)
        if self.args.algorithm != 'fedsgd':
            optimizer.add_param_group(dict(params= list(self.global_model.buffers())))  # add buffered tensors (i.e., gamma and beta of batchnorm layers)
        
        return optimizer
    
    def _create_clients(self, client_datasets):
        CLIENT_CLASS = import_module(f'..client.{self.args.algorithm}client', package= __package__).__dict__[f'{self.args.algorithm.title()}Client']

        def __create_client(identifier, dataset):
            client = CLIENT_CLASS(args= self.args, training_set= dataset[0], test_set= dataset[-1])
            client.id = identifier

            return client
        
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Create clients!')
        clients = []
        with concurrent.futures.ThreadPoolExecutor(max_workers= min(int(self.args.K), os.cpu_count() - 1)) as workhorse:
            for identifier, datasets in TqdmToLogger(
                enumerate(client_datasets),
                logger= logger,
                desc= f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...creating clients...',
                total= len(client_datasets)
            ):
                clients.append(workhorse.submit(__create_client, identifier, datasets).result())
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully created {self.args.K} clients!')

        return clients

    def _sample_clients(self, exclude=[]):
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Sample clients!')
        if exclude == []:   # Update - randomly select max(floor(C*K), 1) clients
            num_sampled_clients = max(int(self.args.C * self.args.K), 1)
            sampled_client_ids = sorted(random.sample([i for i in range(self.args.K)], num_sampled_clients))

        else:   # Evaluation - randomly select unparticipated clients in amount of `eval_fraction` multiplied
            num_unparticipated_clients = self.args.K - len(exclude)
            if num_unparticipated_clients == 0: # when C = 1, i.e., need to evaluate on all clients
                num_sampled_clients = self.args.K
                sampled_client_ids = sorted([i for i in range(self.args.K)])
            
            else:
                num_sampled_clients = max(int(self.args.eval_fraction * num_unparticipated_clients), 1)
                sampled_client_ids = sorted(random.sample([identifier for identifier in [i for i in range(self.args.K)] if identifier not in exclude], num_sampled_clients))
        
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...{num_sampled_clients} clients are selected!')
        
        return sampled_client_ids
    
    def _log_results(self, resulting_sizes, results, eval, participated, save_raw):
        losses, metrics, num_sample = list(), defaultdict(list), list()
        # TODO
