import os 
import glob 
import numpy as np
import torch


class TS_Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, anom_gt_dir=None, ts_split=0.7):
        super().__init__()
        self.data_dir = data_dir
        self.anom_gt_dir = anom_gt_dir
        self.data, self.gt = self.load_data()
        self.ts_split = ts_split

    def __getitem__(self, idx):
        data = np.load(self.data[idx])
        # anom_idx = np.load(self.gt[idx]) if self.gt else None
        # is_anom = anom_idx is not None and len(anom_idx)>0
        data = torch.tensor(data, dtype=torch.float)
        if data.dim() == 1: data = data.unsqueeze(1)
        seq_len = int(data.shape[0]*self.ts_split)
        return data[:seq_len, :], data[seq_len:, :]

    def __len__(self):
        return len(self.data)

    def load_data(self):
        data = glob.glob(os.path.join(self.data_dir, "*.npy"))
        gt = glob.glob(os.path.join(self.anom_gt_dir, "*.npy")) if self.anom_gt_dir is not None else None
        if len(data) == 0:
            raise ValueError("No data found in the specified directory")
        return data, gt