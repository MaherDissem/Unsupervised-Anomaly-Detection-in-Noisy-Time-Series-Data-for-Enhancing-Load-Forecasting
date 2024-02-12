import os
import shutil
import random
import numpy as np 
import torch


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

def delete_files_folders(path):
    """Delete files and folders at the given path."""

    if os.path.isdir(path):    
        os.makedirs(path, exist_ok=True)
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            else:
                shutil.rmtree(item_path)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            os.remove(path)