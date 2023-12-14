import sys
sys.path.append("src") # /src directory
sys.path.append("src/anomaly_detection") # src/AD module

import os
import glob
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import TensorDataset, DataLoader

from anomaly_detection import main as AD_main

# prepare data for AD model training and evaluation
# subprocess call TODO

# train AD model
# subprocess call TODO

# load continuous "load.csv" data
load_serie = pd.read_csv("dataset/processed/AEMO/test/load.csv", index_col=0, parse_dates=True)
gt_serie = pd.read_csv("dataset/processed/AEMO/test/load_gt.csv", index_col=0)

day_size = 48 # same as AD training parameters
n_days = 3
window_size = day_size * n_days
day_stride = 3 # days

# transform into sliding windows
load_windows = []
dates = []
day = 0
while day + window_size < len(load_serie):
    load_windows.append(load_serie.iloc[day: day+window_size].values)
    dates.append(load_serie.index[day: day+window_size])
    day += day_stride*day_size
load_windows = np.array(load_windows)
load_dataset = TensorDataset(torch.tensor(load_windows, dtype=torch.float)) # batch_size x seq_len x 1=feature_dim
infer_dataloader = DataLoader(load_dataset, batch_size=32, shuffle=False) 

# load AD model
default_args = AD_main.parse_args()
coreset = AD_main.get_coreset(default_args, device)
coreset.load_from_path("results/weights", device, AD_main.common.FaissNN(False, 4), )

# infer AD model
scores, heatmaps, labels_gt = coreset.predict(infer_dataloader)
scores = (scores - coreset.min_score) / (coreset.max_score - coreset.min_score + 1e-5)
scores = np.mean(scores, axis=0)

heatmaps = np.array(heatmaps)
heatmaps = (heatmaps - coreset.min_heatmap_scores) / (coreset.max_heatmap_scores - coreset.min_heatmap_scores)
heatmaps = np.mean(heatmaps, axis=-1)

anomaly_free = []
anomalous = []
save_imputation_train_path = "dataset/processed/AEMO/test/ai_train/data"
os.makedirs(save_imputation_train_path, exist_ok=True)

with tqdm.tqdm(infer_dataloader, desc="Saving anomaly free samples to train Imputation model...", leave=True) as data_iterator:
    i = 0 # index of timeserie
    for timeserie_batch in data_iterator:
        timeserie_batch = timeserie_batch[0]
        for timeserie in timeserie_batch:
            date_range = dates[i]
            score = scores[i]
            heatmap = heatmaps[i]
            if score<=coreset.window_threshold:
                anomaly_free.append((timeserie, date_range))

            else:
                highest_patch_score = np.argmax(heatmap) # if this doest work, simply take the patch containing max / min or compare with mean
                patch_size = timeserie.shape[0] // heatmap.shape[0]
                patch_start = highest_patch_score * patch_size
                patch_end = patch_start + patch_size
                masked_data = timeserie.clone()
                masked_data[patch_start: patch_end] = 0
                anomalous.append((masked_data, date_range))

                # save heatmap jpg
                # heatmap = heatmaps[i].reshape(1, -1)
                # heatmap_data = np.repeat(heatmap, len(timeserie)//heatmap.shape[1], axis=1)
                # fig, ax1 = plt.subplots(figsize=(10, 6))
                # ax2 = ax1.twinx()
                # ax1.imshow(heatmap_data, cmap="YlOrRd", aspect='auto')
                # ax2.plot(timeserie, label='Time Series', color='blue')
                # ax2.set_xlabel('Time')
                # ax2.set_ylabel('Value', color='blue')
                # ax2.tick_params('y', colors='blue')
                # plt.title('Time Series with Anomaly Score Heatmap')
                # os.makedirs("out/anom/heatmaps", exist_ok=True, )
                # plt.savefig(f"out/anom/heatmaps/{i}.png")
                # plt.close()
                
            i += 1

# save anomaly free samples to train Imputation model
# transform to single day samples
anomaly_free_days = []
for timeserie, date_range in anomaly_free:
    for i in range(0, len(timeserie), day_size):
        day_serie = timeserie[i:i+day_size].squeeze(-1)
        date = date_range[i].date()
        anomaly_free_days.append(day_serie)
        np.save(os.path.join(save_imputation_train_path, str(date)+'.npy'), day_serie)

# train anomaly imputation model on anomaly free samples
# subprocess call TODO

# infer anomaly detection and impute anomalies on anomalous samples
from anomaly_imputation import main as AI_main
default_args = AI_main.parse_args()

# anomaly imputation should run once, than infer AD again and impute anomalies again


