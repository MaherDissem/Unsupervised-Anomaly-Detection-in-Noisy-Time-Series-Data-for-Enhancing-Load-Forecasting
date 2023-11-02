import os 
import glob 
import numpy as np
import torch


class TS_Dataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, ts_split=0.7):
        super().__init__()
        self.root_dir = root_dir
        # self.data, self.gt = self.load_data()
        self.data = self.load_data()
        self.ts_split = ts_split

    def __getitem__(self, idx):
        data = np.load(self.data[idx])
        data = torch.tensor(data, dtype=torch.float)
        # anom_idx = np.load(self.gt[idx])
        # is_anom = anom_idx is not None and anom_idx.sum()>0
        if data.dim() == 1: data = data.unsqueeze(1)
        seq_len = int(data.shape[0]*self.ts_split)
        return data[:seq_len, :], data[seq_len:, :]

    def __len__(self):
        return len(self.data)

    def load_data(self):
        data = glob.glob(os.path.join(self.root_dir, "data", "*.npy"))
        # gt = glob.glob(os.path.join(self.root_dir, "gt", "*.npy"))
        if len(data) == 0: 
            raise ValueError("No data found in the specified directory")
        return data