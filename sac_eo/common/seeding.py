"""Helper functions to set random seeds."""
import numpy as np
import tensorflow as tf
import random
import os

def init_seeds(seed,env=None):
    """Sets random seed."""
    seed = int(seed)
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)