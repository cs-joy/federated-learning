import logging
import numpy as np

from src import TqdmToLogger

logger = logging.getLogger(__name__)



def simulate_split(args, dataset):
    """
    Split data indices using labels.

    Args:
        args (argsparser): arguments
        dataset (dataset): raw dataset instance to be split
    
    Returns:
        split_map (dict): dictionary with key is a client index and a corresponding value is a list of indices
    """
    pass