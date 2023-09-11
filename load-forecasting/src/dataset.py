import os 
import glob 
import numpy as np
import torch


class TS_Dataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, data_type, ts_split):
        super().__init__()
        self.root_dir = root_dir
        self.data_type = data_type
        self.data, self.gt = self.load_data()
        self.ts_split = ts_split

    def __getitem__(self, idx):
        data = np.load(self.data[idx])
        # anom_idx = np.load(self.gt[idx]) if self.gt else None
        # is_anom = anom_idx is not None and len(anom_idx)>0
        data = torch.tensor(data, dtype=torch.float)
        if self.data_type == "contam":
            data = data.unsqueeze(1)
        seq_len = int(data.shape[0]*self.ts_split)
        return data[:seq_len, :], data[seq_len:, :]

    def __len__(self):
        return len(self.data)

    def load_data(self):
        if self.data_type == "filter":
            data = glob.glob(os.path.join(self.root_dir, "test", "data", "*.npy")) # TODO give path directly instead of joining
            gt = glob.glob(os.path.join(self.root_dir, "test", "gt", "*.npy"))
        if self.data_type == "contam":
            data = glob.glob(os.path.join(self.root_dir, "test", "data", "*.npy"))
            gt = glob.glob(os.path.join(self.root_dir, "test", "gt", "*.npy"))
        if len(data) == 0:
            raise ValueError("No data found in the specified directory")
        return data, gt