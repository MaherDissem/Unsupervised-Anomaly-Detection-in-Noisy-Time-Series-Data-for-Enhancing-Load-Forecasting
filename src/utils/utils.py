import random
import numpy as np 
import torch
import os 
import glob


def set_seed(seed=0, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def make_clean_folder(path):
    os.makedirs(path, exist_ok=True)
    for f in glob.iglob(os.path.join(path, "*")):
        os.remove(f)