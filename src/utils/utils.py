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

def find_consec_values(lst, min_consecutive=3, indices_only=True, pad=2):
    """Find consecutive values in a list and return their indices with optional padding."""
    out_map = []
    out_indices = []
    current_value = None
    current_indices = []

    for i, value in enumerate(lst):
        if value == current_value:
            current_indices.append(i)
        else:
            if len(current_indices) >= min_consecutive:
                # Add padding to the left and right of consecutive values interval
                start_index = max(0, current_indices[0] - pad)
                end_index = min(len(lst) - 1, current_indices[-1] + pad)
                out_map.append((current_value, list(range(start_index, end_index + 1))))
                out_indices.extend(range(start_index, end_index + 1))
            current_value = value
            current_indices = [i]

    # Check for consecutive values at the end of the list
    if len(current_indices) >= min_consecutive:
        # Add padding to the left and right of consecutive values interval
        start_index = max(0, current_indices[0] - pad)
        end_index = min(len(lst) - 1, current_indices[-1] + pad)
        out_map.append((current_value, list(range(start_index, end_index + 1))))
        out_indices.extend(range(start_index, end_index + 1))

    if indices_only:
        return out_indices

    return out_map

def make_clean_folder(path):
    os.makedirs(path, exist_ok=True)
    for f in glob.iglob(os.path.join(path, "*")):
        os.remove(f)