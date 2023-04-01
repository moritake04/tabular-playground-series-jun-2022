import random

import numpy as np

# import torch


def seed_everything(seed=42):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    """
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    """
    print("seed set to {}".format(seed))


"""
def get_device(gpu_id=-1):
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device("cuda", gpu_id)
    else:
        device = torch.device(0)
    print("using {}".format(device))
    return device
"""
