import os
import glob
import numpy as np
import torch


class DatasetAnomalyDetection(torch.utils.data.Dataset):

    def __init__(self, data_folders_paths):
        super().__init__()
        self.data_folders_paths = data_folders_paths
        self.data, self.gt = self.load_data()

    def __getitem__(self, idx):
        data = np.load(self.data[idx])
        gt_heatmap = np.load(self.gt[idx]) if len(self.gt) else np.zeros_like(data)
        is_anom = np.any(gt_heatmap)
        return {
            "data": torch.tensor(data, dtype=torch.float).unsqueeze(1),
            "gt_heatmap": gt_heatmap,
            "is_anomaly": is_anom
        }

    def __len__(self):
        return len(self.data)

    def load_data(self):
        data = []
        gt = []
        for root_dir in self.data_folders_paths:
            data.extend(glob.glob(os.path.join(root_dir, "data", "*.npy")))
            gt.extend(glob.glob(os.path.join(root_dir, "gt", "*.npy")))
        if len(data) == 0: 
            raise ValueError("No data found in the specified directory")
        return data, gt
