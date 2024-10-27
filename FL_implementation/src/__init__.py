import logging
import transformers

# turn off unnecessary logging
transformers.logging.set_verbosity_error()

from .utils import set_seed, Range, TensorBoardRunner, check_args, init_weights, TqdmToLogger, MetricManager, stratified_split
from .loaders import load_dataset, load_model



# for logger initialization
def set_logger(path, args):
    pass