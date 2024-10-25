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