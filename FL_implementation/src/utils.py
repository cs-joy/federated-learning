import os
import sys
import json
import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from importlib import import_module
from collections import defaultdict
from multiprocessing import Process


logger = logging.getLogger(__name__)



###########################
#       tqdm add-on       #
###########################
class TqdmToLogger(tqdm):
    def __init__(self, *args, logger=None,
        min_interval = 0.1,
        bar_format = '{desc:<}{percentage: 3.0f}% |{bar: 20}| [{n_fmt: 6s}/{total_fmt}]',
        desc = None,
        **kwargs
    ):
        self._logger = logger
        super().__init__(*args, mininterval=min_interval, bar_format=bar_format, desc=desc, **kwargs)
    
    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        return logger
    
    def display(self, msg=None, pos= None):
        if not self.n:
            return
        if not msg:
            msg = self.__str__()
        self.logger.info('%s', msg.strip('\r\n\t '))


##########################
#  Weight Initialization #
##########################
def init_weights(model, init_type, init_gain):
    """
    Initialize network weights.

    Args:
        model (torch.nn.Module): network to be initialized
        init_type (string): the name of an initialization method: normal | xavier | xavier_uniform | kaiming | truncnorm | orthogonal | none
        init_gain (float): scaling factor for normal, xavier and orthogonal
    
    Returns:
        model (torch.nn.Module): initialized model with `init_type` and `init_gain`
    """
    def init_func(m):   # define the initializtion function
        class_name = m.__class__.__name__
        if class_name.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, mean= 1.0, std= init_gain)

            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (class_name.find('Linear') == 0 or class_name.find('Conv') == 0):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, mean= 0, std= init_gain)
            
            elif init_type == 'xavier':
                torch.nn.inti.xavier_normal_(m.weight.data, gain= init_gain)
            
            elif init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight.data, gain= init_gain)
            
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a= 0, mode= 'fan_in')
            
            elif init_type == 'truncnorm':
                torch.nn.init.trunc_normal_(m.weight.data, mean= 0., std= init_gain)
            
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain= init_gain)
            
            elif init_type == 'none':   # uses pytorch's default init method
                m.reset_parameters()
            
            else:
                raise NotImplementedError(f'[ERROR] Initialization method {init_type} is not implemented!')
            
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    model.apply(init_func)


###########################
#      Metric Manager     #
###########################
class MetricManager:
    """
    Managing metrics to be used
    """
    def __init__(self, eval_metrics):
        self.metric_funcs = {
            name: import_module(f'.metrics', package= __package__).__dict__[name.title()]()
            for name in eval_metrics
        }
        self.figures = defaultdict(int)
        self._results = dict()

        # use optimal threshold (i.e., Youden's J or not)
        if 'youdenj' in self.metric_funcs:
            for func in self.metric_funcs.values():
                if hasattr(func, '_use_youdenj'):
                    setattr(func, '_use_youdenj', True)
    
    def track(self, loss, pred, true):
        # update running loss
        self.figures['loss'] += loss * len(pred)

        # update running metrics
        for module in self.metric_funcs.values():
            module.collect(pred, true)

    def aggregate(self, total_len, curr_step= None):
        running_figures = {name: module.summarize() for name, module in self.metric_funcs.items()}
        running_figures['loss'] = self.figures['loss'] / total_len
        if curr_step is not None:
            self._results[curr_step] = {
                'loss': running_figures['loss'],
                'metrics': {name: running_figures[name] for name in self.metric_funcs.keys()}
            }
        else:
            self._results = {
                'loss': running_figures['loss'],
                'metrics': {name: running_figures[name] for name in self.metric_funcs.keys()}
            }
        self.figures = defaultdict()
    
    @property
    def results(self):
        return self._results