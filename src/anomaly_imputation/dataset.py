import os
import glob
import numpy as np
import torch


class AI_Dataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, is_train=True, len_mask=8):
        super().__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.len_mask = len_mask
        self.data = self.load_data()

    def __getitem__(self, idx):
        ts = torch.tensor(np.load(self.data[idx]), dtype=torch.float)
        if self.is_train:
            mask = torch.ones_like(ts)
            mask_idx = np.random.randint(0, len(ts) - self.len_mask-1)
            mask[mask_idx: mask_idx+self.len_mask] = 0
            masked_ts = ts * mask
            return {
                "clean_data": ts.unsqueeze(-1),
                "masked_data": masked_ts.unsqueeze(-1),
                "mask": mask.unsqueeze(-1)
            }
        else:
            raise NotImplementedError # we only need masked data for now

    def __len__(self):
        return len(self.data)

    def load_data(self):
        if self.is_train:
            # train model by randomly removing self.len_mask consecutive values from each time series
            data = glob.glob(os.path.join(self.root_dir, "data", "*.npy"))
        else: # inference on whole dataset at once 
            raise NotImplementedError            
        return data
    