import glob
import os
import numpy as np
import torch


class TS_Dataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, data_type):
        super().__init__()
        self.root_dir = root_dir
        self.data_type = data_type
        self.data, self.gt = self.load_data()

    def __getitem__(self, idx):
        data = np.load(self.data[idx])
        anom_idx = np.load(self.gt[idx]) if self.gt else None
        is_anom = anom_idx is not None and len(anom_idx)>0
        return {
            "data": torch.tensor(data, dtype=torch.float).unsqueeze(1),
            # "anom_idx": anom_idx if np.any(anom_idx) else [], # define collate_func to handle lists of different sizes
            "is_anomaly": is_anom
        }

    def __len__(self):
        return 100 # hack to make pipeline testing faster, remove later
        return len(self.data)

    def load_data(self):
        if self.data_type == "test":
            data = glob.glob(os.path.join(self.root_dir, "test", "data", "*.npy"))
            gt = glob.glob(os.path.join(self.root_dir, "test", "gt", "*.npy"))
        if self.data_type == "train":
            data = glob.glob(os.path.join(self.root_dir, "train", "data", "*.npy"))
            gt = glob.glob(os.path.join(self.root_dir, "train", "gt", "*.npy"))
        if len(data) == 0:
            raise ValueError("No data found in the specified directory")
        return data, gt
