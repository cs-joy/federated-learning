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
        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [CLIENT] < {str(identifier).zfill(6)} > '
            if eval:    # get loss and metrics
                # loss
                loss = result['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)

                # metrics
                for metric, value in result['metrics'].items():
                    client_log_string += f'| {metrics}: {value:.4f} '
                    metrics[metric].append(value)

            else:   # same, but retrieve results of last epoch's
                # loss
                loss = results[self.args.E]['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
            
                # metrics
                for name, value in result[self.args.E]['metrics'].items():
                    client_log_string += f'| {name}: {value:.4f} '
                    metrics[name].append(value)
            
            # get sample size
            num_sample.append(resulting_sizes[identifier])

            # log per client
            logger.info(client_log_string)
        
        else:
            num_samples = np.array(num_samples).astype(float)
        
        # aggregate into total logs
        result_dict = defaultdict(dict)
        total_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [SUMMARY] ({len(resulting_sizes)} clients):'

        # loss
        losses_array = np.array(losses).astype(float)
        weighted = losses_array.dot(num_samples) / sum(num_samples); std = losses_array.std()

        top10_indices = np.argpartition(losses_array, -int(0.1 * len(losses_array)))[-int(0.1 * len(losses_array)):] if len(losses_array) > 1 else 0
        top10 = np.atleast_1d(losses_array[top10_indices])
        top10_mean, top10_std = top10.dot(np.atleast_1d(num_samples[top10_indices])) / num_samples[top10_indices].sum(), top10.std()

        bot10_indices = np.argpartition(losses_array, max(1, int(0.1 * len(losses_array)) - 1))[:max(1, int(0.1 * len(losses_array)))] if len(losses_array) > 1 else 0
        bot10 = np.atleast_1d(losses_array[bot10_indices])
        bot10_mean, bot10_std = bot10.dot(np.atleast_1d(num_samples[bot10_indices])) / num_samples[bot10_indices].sum(), bot10.std()

        total_log_string += f'\n    - Loss: Avg. ({weighted:.4f}) Std. ({std:.4f}) | Top 10% ({top10_mean:.4f}) Std. ({top10_std:.4f}) | Bottom 10% ({bot10_mean:.4f}) Std. ({bot10_std})'
        result_dict['loss'] = {
            'avg': weighted.astype(float), 'std': std.astype(float),
            'top10p_avg': top10_mean.astype(float), 'top10p_std': top10_std.astype(float),
            'bottom10p_avg': bot10_mean.astype(float), 'bottom10p_std': bot10_std.astype(float)
        }

        if save_raw:
            result_dict['loss']['raw'] = losses

        self.writer.add_scalars(
            f'Local {"Test" if eval else "Training"} Loss' + eval * f'({"In" if participated else "Out"})',
            {'Avg.': weighted, 'Std.': std, 'Top 10% Avg.': top10_mean, 'Top 10% Std.': top10_std, 'Bottom 10% Avg.': bot10_mean, 'Bottom 10% Std.': bot10_std},
            self.round
        )

        # metrics
        for name, val in metrics.items():
            val_array = np.array(val).astype(float)
            weighted = val_array.dot(num_samples) / sum(num_samples); std = val_array.std()

            top10_indices = np.argpartition(val_array, -int(0.1 * len(val_array)))[-int(0.1 * len(val_array)):] if len(val_array) > 1 else 0
            top10 = np.atleast_1d(val_array[top10_indices])
            top10_mean, top10_std = top10.dot(np.atleast_1d(num_samples[top10_indices])) / num_samples[top10_indices].sum(), top10.std()

            bot10_indices = np.argpartition(val_array, max(1, int(0.1 * len(val_array)) - 1))[:max(1, int(0.1 * len(val_array)))] if len(val_array) > 1 else 0
            bot10 = np.atleast_1d(val_array[bot10_indices])
            bot10_mean, bot10_std = bot10.dot(np.atleast_1d(num_samples[bot10_indices])) / num_samples[bot10_indices].sum(), bot10.std()

            total_log_string += f'\n    - {name.title()}: Avg. ({weighted:.4f}) Std. ({std:.4f}) | Top 10% ({top10_mean:.4f}) Std. ({top10_std:.4f}) | Bottom 10% ({bot10_mean:.4f}) Std. ({bot10_std:.4f})'
            result_dict[name] = {
                'avg': weighted.astype(float), 'std': std.astype(float),
                'top10p_avg': top10_mean.astype(float), 'top10p_std': top10_std.astype(float),
                'bottom10p_avg': bot10_mean.astype(float), 'bottom10p_std': bot10_std.astype(float)
            }

            if save_raw:
                result_dict[name]['raw'] = val
            
            self.writer.add_scalars(
                f'Local {"Test" if eval else "Training"} {name.title()}' + eval * f' ({"In" if participated else "Out"})',
                {'Avg.': weighted, 'Std.': std, 'Top 10% Avg.': top10_mean, 'Top 10% Std.': top10_std, 'Bottom 10% Avg.': bot10_mean, 'Bottom 10% Std.': bot10_std},
                self.round
            )
            self.writer.flush()
        
        # log total message
        logger.info(total_log_string)
        
        return result_dict
