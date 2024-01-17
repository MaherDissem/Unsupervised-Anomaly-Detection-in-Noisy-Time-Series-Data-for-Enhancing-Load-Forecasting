import os
import glob
import numpy as np
import torch


class DatasetAnomalyImputation(torch.utils.data.Dataset):

    def __init__(self, root_dir, mask_size=8):
        super().__init__()
        self.root_dir = root_dir
        self.mask_size = mask_size
        self.data = self.load_data()

    def __getitem__(self, idx):
        ts = torch.tensor(np.load(self.data[idx]), dtype=torch.float)
        mask = torch.ones_like(ts)
        mask_idx = np.random.randint(0, len(ts) - self.mask_size-1)
        mask[mask_idx: mask_idx+self.mask_size] = 0
        masked_ts = ts * mask
        return {
            "clean_data": ts.unsqueeze(-1),
            "masked_data": masked_ts.unsqueeze(-1),
            "mask": mask.unsqueeze(-1)
        }

    def __len__(self):
        return len(self.data)

    def load_data(self):
        data = glob.glob(os.path.join(self.root_dir, "*.npy"))
        return data
    