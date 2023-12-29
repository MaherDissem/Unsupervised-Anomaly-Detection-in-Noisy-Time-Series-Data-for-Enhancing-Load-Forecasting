import torch 
import numpy as np


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


def heatmap_postprocess(timeserie, heatmap, flag_highest_patch=True, extend_to_patch=True, anom_idx_only=False):
    anom_idx = []
    patch_size = timeserie.shape[0] // heatmap.shape[0]

    # high anomaly score patch
    if flag_highest_patch: # this part is disbled for now, as it is not precise enough
        highest_patch_score_idx = np.argmax(heatmap) 
        patch_start = highest_patch_score_idx * patch_size 
        patch_end = patch_start + patch_size
        anom_idx = list(range(patch_start, patch_end))

    # consecutive values, for when power drops to 0 before spike (anomaly type 2)
    anom_idx += find_consec_values(timeserie, min_consecutive=2, indices_only=True, pad=patch_size//2)

    # outliers in anomalous samples
    timeserie_ = timeserie.clone()
    timeserie_ = (timeserie_ - timeserie_.mean()) / timeserie_.std()
    spike_anom_idx = torch.nonzero(timeserie_ > 2.5*timeserie_.std())[:, 0].tolist()
    spike_anom_idx += torch.nonzero(timeserie_ < -2.5*timeserie_.std())[:, 0].tolist()

    # extend point to patch for smoother imputation
    if extend_to_patch:
        for point in spike_anom_idx:
            if point not in anom_idx:
                anom_idx += list(range(max(0, point-patch_size//2), min(point+patch_size//2+1, len(timeserie)-1)))             

    anom_idx = list(set(anom_idx)) 

    if anom_idx_only:
        return anom_idx

    mask = torch.zeros_like(timeserie)
    mask[anom_idx] = 1
    return mask

