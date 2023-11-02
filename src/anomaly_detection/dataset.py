import glob
import os
import numpy as np
import torch


class TS_Dataset(torch.utils.data.Dataset):

    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.data, self.gt = self.load_data()

    def __getitem__(self, idx):
        data = np.load(self.data[idx])
        anom_idx = np.load(self.gt[idx])
        is_anom = anom_idx is not None and anom_idx.sum()>0
        return {
            "data": torch.tensor(data, dtype=torch.float).unsqueeze(1),
            # "anom_idx": anom_idx if np.any(anom_idx) else [], # define collate_func to handle lists of different sizes
            "is_anomaly": is_anom
        }

    def __len__(self):
        return len(self.data)
        return 1000 # hack to make pipeline testing faster, remove later

    def load_data(self):
        data = glob.glob(os.path.join(self.root_dir, "data", "*.npy"))
        gt = glob.glob(os.path.join(self.root_dir, "gt", "*.npy"))
        if len(data) == 0: 
            raise ValueError("No data found in the specified directory")
        return data, gt
