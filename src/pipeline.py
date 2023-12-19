import sys
sys.path.append("src") # /src directory
sys.path.append("src/anomaly_detection") # src/AD module
sys.path.append("src/anomaly_imputation") # src/AI module

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
from utils.utils import find_consec_values
from utils.utils import make_clean_folder
from utils.utils import set_seed
set_seed(0)

# prepare data for AD model training and evaluation
# subprocess call TODO

# train AD model
# subprocess call TODO

# load continuous "load.csv" data
load_serie = pd.read_csv("dataset/processed/AEMO/test/load.csv", index_col=0, parse_dates=True)
gt_serie = pd.read_csv("dataset/processed/AEMO/test/load_gt.csv", index_col=0)

day_size = 48 # same as AD training parameters
n_days = 1
window_size = day_size * n_days
day_stride = 1 # days

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

save_imputation_train_path = "dataset/processed/AEMO/test/ai_train/data"
path_list = [save_imputation_train_path, "results/heatmaps", "results/imputation", ]
for path in path_list:
    make_clean_folder(path)

anomaly_free = []
anomalous = []

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
                # TODO save anomaly free samples to train Imputation model HERE

            else:
                # spikes are detected using patch anomaly score heatmap and outlier rules
                # anomaly type 2 replaces values by 0 before spike, mask should also account for this TODO

                # high anomaly score patch
                highest_patch_score_idx = np.argmax(heatmap) 
                patch_size = timeserie.shape[0] // heatmap.shape[0] 
                patch_start = highest_patch_score_idx * patch_size 
                patch_end = patch_start + patch_size
                anom_idx = list(range(patch_start, patch_end+1))

                # consecutive values
                anom_idx += find_consec_values(timeserie, min_consecutive=2, indices_only=True, pad=patch_size//2)

                # outliers
                timeserie_ = timeserie.clone()
                timeserie_ = (timeserie_ - timeserie_.mean()) / timeserie_.std()
                spike_anom_idx = torch.nonzero(timeserie_ > 2.5*timeserie_.std())[:, 0].tolist()
                spike_anom_idx += torch.nonzero(timeserie_ < -2.5*timeserie_.std())[:, 0].tolist()
                # extend point to patch for smoother imputation
                for point in spike_anom_idx:
                    if point not in anom_idx:
                        anom_idx += list(range(max(0, point-patch_size//2), min(point+patch_size//2+1, len(timeserie))))             

                anom_idx = list(set(anom_idx)) 
                masked_data = timeserie.clone()
                masked_data[anom_idx] = 0
                mask = torch.ones_like(masked_data)
                mask[anom_idx] = 0 
                anomalous.append((masked_data, mask, date_range))

                # save heatmap jpg
                heatmap = heatmaps[i].reshape(1, -1)
                heatmap_data = np.repeat(heatmap, len(timeserie)//heatmap.shape[1], axis=1)
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax2 = ax1.twinx()
                ax1.imshow(heatmap_data, cmap="YlOrRd", aspect='auto')
                ax2.plot(timeserie, label='Time Series', color='blue')
                ax2.plot(mask, label='Mask', color='green')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Value', color='blue')
                ax2.tick_params('y', colors='blue')
                plt.title('Time Series with Anomaly Score Heatmap')
                plt.legend()
                os.makedirs("results/heatmaps", exist_ok=True, )
                plt.savefig(f"results/heatmaps/{i}.png")
                plt.close()

            i += 1

# transform to single day samples (testing if multiple day AD is easier than single day AD)
# anomalous_data_1day = []
# anomaly_free_data_1day = []

# for masked_data, mask, date_range in anomalous:
#     for i in range(0, len(masked_data), day_size):
#         day_serie = masked_data[i:i+day_size].unsqueeze(0)
#         day_mask = mask[i:i+day_size]
#         date = date_range[i].date()
#         if day_mask[day_mask==0].shape[0] == 0:
#             anomaly_free_data_1day.append((day_serie, date))
#         else:
#             anomalous_data_1day.append((day_serie, day_mask, date))

# for timeserie, date_range in anomaly_free:
#     for i in range(0, len(timeserie), day_size):
#         day_serie = timeserie[i:i+day_size].squeeze(-1)
#         date = date_range[i].date()
#         anomaly_free_data_1day.append((day_serie, date))

#         # save anomaly free samples to train Imputation model
#         np.save(os.path.join(save_imputation_train_path, str(date)+'.npy'), day_serie)

for timeserie, date_range in anomaly_free:
    # save anomaly free samples to train Imputation model
    date = str(date_range[0].date()) + "_" + str(date_range[-1].date())
    dates = ''
    for d in range(n_days):
        dates += str(date_range[d*day_size].date()) + "_"
    np.save(os.path.join(save_imputation_train_path, str(dates)+'.npy'), timeserie.squeeze(-1))
print(f"saved impuation plots to {save_imputation_train_path}.npy")
# exit(0)

# train anomaly imputation model on anomaly free samples
from anomaly_imputation import train as AI_train
trained = False
if not trained:
    AI_train.train(AI_train.parse_args())

# infer anomaly detection and impute anomalies on anomalous samples
from anomaly_imputation.model import LSTM_AE
from anomaly_imputation.train import parse_args as AI_train_parse_args

default_args = AI_train_parse_args()
loaded_model = LSTM_AE(default_args.seq_len, default_args.no_features, default_args.embedding_dim, default_args.learning_rate, default_args.every_epoch_print, default_args.epochs, default_args.patience, default_args.max_grad_norm)
loaded_model.load()

save_imputation_path = os.path.join("results", "imputation")
os.makedirs(save_imputation_path, exist_ok=True)
for f in glob.glob(os.path.join(save_imputation_path, "*")):
    os.remove(f)

for i, (masked_data, mask, date_range) in enumerate(anomalous):
    filled_ts = loaded_model.impute(masked_data.unsqueeze(0), mask)

    plt.plot(masked_data.squeeze(0).squeeze(-1), label="serie with missing values")
    plt.plot(mask.squeeze(0).squeeze(-1), label="mask")
    plt.plot(filled_ts, label="serie with filled values")
    plt.legend()
    plt.savefig(os.path.join(save_imputation_path, f"{i}.png"))
    plt.clf()


