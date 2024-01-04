import os 
import glob 
import numpy as np
import torch
import pandas as pd

class F_Dataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, ts_split=0.7):
        super().__init__()
        self.root_dir = root_dir
        self.npy_paths = self.load_data()
        self.ts_split = ts_split

    def __getitem__(self, idx):
        date_range = os.path.basename(self.npy_paths[idx]).replace(".npy", "")
        first_date, last_date = date_range.split(" - ")
        first_date = first_date.split("_")[0]
        last_date = last_date.split("_")[0]
        dates = pd.date_range(first_date, last_date, freq="1D")
        dates = [str(date).split(" ")[0] for date in dates]

        data = np.load(self.npy_paths[idx])
        data = torch.tensor(data, dtype=torch.float)
        if data.dim() == 1: data = data.unsqueeze(1)
        seq_len = int(data.shape[0]*self.ts_split)
        return dates, data[:seq_len, :], data[seq_len:, :]

    def __len__(self):
        return len(self.npy_paths)

    def load_data(self):
        npy_paths = glob.glob(os.path.join(self.root_dir, "data", "*.npy"))
        if not len(npy_paths): 
            raise ValueError("No data found in the specified directory")
        return npy_paths
