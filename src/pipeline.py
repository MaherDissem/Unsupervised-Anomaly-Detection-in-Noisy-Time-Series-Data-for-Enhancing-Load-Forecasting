import sys
sys.path.append("src") # /src directory
sys.path.append("src/anomaly_detection") # AD module
sys.path.append("src/anomaly_imputation") # AI module

import os
import glob
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

from anomaly_detection import main as AD_main
from anomaly_detection.postprocessing import heatmap_postprocess
from utils.utils import find_consec_values
from utils.utils import make_clean_folder
from utils.utils import set_seed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0)

# parameters
data_folder = "AEMO/NSW"
day_size = 48
n_days = 1
window_size = day_size * n_days
day_stride = 1 # days
contam_ratio = 0.1
forecast_window_size = 5
save_figs = False
imp_trained = True

# ---
# Generate synthetic data
# ---
from data.prepare_data_AD import run as prepare_data_AD_run
from data.prepare_data_AD import parse_args as prepare_data_AD_parse_args

default_prepare_data_AD_args = prepare_data_AD_parse_args()
default_prepare_data_AD_args.raw_data_csv = raw_data_root = f"dataset/{data_folder}"
default_prepare_data_AD_args.trg_save_data = f"dataset/processed/{data_folder}"
default_prepare_data_AD_args.day_size = day_size
default_prepare_data_AD_args.n_days = n_days
default_prepare_data_AD_args.day_stride = day_stride

prepare_data_AD_run(default_prepare_data_AD_args)

# ---
# train AD model
# ---
from anomaly_detection.main import run as AD_run
from anomaly_detection.main import parse_args as AD_parse_args

default_AD_args = AD_parse_args()
default_AD_args.train_data_path = [f"dataset/processed/{data_folder}/ad_train_contam", f"dataset/processed/{data_folder}/ad_test_contam"]
default_AD_args.test_data_path = [f"dataset/processed/{data_folder}/ad_train_contam", f"dataset/processed/{data_folder}/ad_test_contam"]
default_AD_args.nbr_timesteps = window_size
default_AD_args.contam_ratio = contam_ratio

AD_run(default_AD_args)

# ---
# find and impute anomalies
# ---

# load continuous "load.csv" data
load_serie = pd.read_csv(f"dataset/processed/{data_folder}/load_contam.csv", index_col=0, parse_dates=True)
gt_serie = pd.read_csv(f"dataset/processed/{data_folder}/load_contam_gt.csv", index_col=0)

# transform into sliding windows (same parameters as AD training)
def sliding_windows(load_serie, window_size, day_stride, day_size):
    load_windows = []
    dates = []
    day = 0
    while day + window_size < len(load_serie):
        load_windows.append(load_serie.iloc[day: day+window_size].values)
        dates.append(load_serie.index[day: day+window_size])
        day += day_stride*day_size
    return load_windows, dates

load_windows, dates = sliding_windows(load_serie, window_size, day_stride, day_size)
load_windows = np.array(load_windows)
load_dataset = TensorDataset(torch.tensor(load_windows, dtype=torch.float)) # batch_size x seq_len x 1=feature_dim
infer_dataloader = DataLoader(load_dataset, batch_size=32, shuffle=False) # GT not needed

# load AD model
default_args = AD_main.parse_args()
coreset = AD_main.get_coreset(default_args, device)
coreset.load_from_path("results/weights", device, AD_main.common.FaissNN(False, 4))

# infer AD model
scores, heatmaps, _, _, _ = coreset.predict(infer_dataloader)
scores = (scores - coreset.min_score) / (coreset.max_score - coreset.min_score + 1e-5)
scores = np.mean(scores, axis=0)

heatmaps = np.array(heatmaps)
heatmaps = (heatmaps - coreset.min_heatmap_scores) / (coreset.max_heatmap_scores - coreset.min_heatmap_scores)
heatmaps = np.mean(heatmaps, axis=-1)

save_imputation_train_path = f"dataset/processed/{data_folder}/ai_train/data"
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
                anomaly_free.append((timeserie, date_range)) # each timeseries[k] has a date_range[k] associated
                # TODO save anomaly free samples to train Imputation model HERE

            else:
                # high anomaly score patch
                highest_patch_score_idx = np.argmax(heatmap) 
                patch_size = timeserie.shape[0] // heatmap.shape[0] 
                patch_start = highest_patch_score_idx * patch_size 
                patch_end = patch_start + patch_size
                anom_idx = list(range(patch_start, patch_end))
                # consecutive values, anomaly type 2 replaces values by 0 before spike
                anom_idx += find_consec_values(timeserie, min_consecutive=2, indices_only=True, pad=patch_size//2)
                # outliers
                timeserie_ = timeserie.clone()
                timeserie_ = (timeserie_ - timeserie_.mean()) / timeserie_.std()
                spike_anom_idx = torch.nonzero(timeserie_ > 2.5*timeserie_.std())[:, 0].tolist()
                spike_anom_idx += torch.nonzero(timeserie_ < -2.5*timeserie_.std())[:, 0].tolist()
                # extend point to patch for smoother imputation
                for point in spike_anom_idx:
                    if point not in anom_idx:
                        anom_idx += list(range(max(0, point-patch_size//2), min(point+patch_size//2+1, len(timeserie)-1)))             

                anom_idx = list(set(anom_idx)) 
                masked_data = timeserie.clone()
                masked_data[anom_idx] = 0
                mask = torch.ones_like(masked_data)
                mask[anom_idx] = 0 
                anomalous.append((masked_data, mask, date_range))

                if save_figs:
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

# save anomaly free samples to train Imputation model
for timeserie, date_range in anomaly_free:
    date = str(date_range[0].date()) + "_" + str(date_range[-1].date())
    dates = ''
    for d in range(n_days):
        dates += str(date_range[d*day_size].date()) + "_"
    np.save(os.path.join(save_imputation_train_path, str(dates)+'.npy'), timeserie.squeeze(-1))
print(f"saved impuation plots to {save_imputation_train_path}.npy")

# train anomaly imputation model on anomaly free samples
from anomaly_imputation.train import parse_args as AI_parse_args
from anomaly_imputation.train import train as AI_train
from anomaly_imputation.model import LSTM_AE

default_AI_args = AI_parse_args()
default_AI_args.seq_len = window_size
default_AI_args.mask_size = patch_size
default_AI_args.dataset_root = save_imputation_train_path

if not imp_trained:
    AI_train(default_AI_args)

# infer anomaly detection and impute anomalies on anomalous samples
loaded_model = LSTM_AE(default_AI_args.seq_len, default_AI_args.no_features, default_AI_args.embedding_dim, default_AI_args.learning_rate, default_AI_args.every_epoch_print, default_AI_args.epochs, default_AI_args.patience, default_AI_args.max_grad_norm)
loaded_model.load()

save_imputation_path = os.path.join("results", "imputation")
os.makedirs(save_imputation_path, exist_ok=True)
for f in glob.glob(os.path.join(save_imputation_path, "*")):
    os.remove(f)

cleaned_anomalies = []
for i, (masked_data, mask, date_range) in enumerate(anomalous):
    filled_ts = loaded_model.impute(masked_data.unsqueeze(0), mask)
    cleaned_anomalies.append((filled_ts, date_range))

    if save_figs:
        # save imputation visualization
        plt.plot(masked_data.squeeze(0).squeeze(-1), label="serie with missing values")
        plt.plot(mask.squeeze(0).squeeze(-1), label="mask")
        plt.plot(filled_ts, label="serie with filled values")
        plt.legend()
        plt.savefig(os.path.join(save_imputation_path, f"{i}.png"))
        plt.clf()


# ---
# save cleaned (imputed) data for forecasting model
# ---
# we reconstruct a continuous timeserie from individually imputed windows          
  
cleaned_dataset = []
for timeserie, date_range in cleaned_anomalies:
    for i in range(0, len(timeserie), day_size):
        day_serie = timeserie[i:i+day_size]
        date = date_range
        cleaned_dataset.append((day_serie, date))

for timeserie, date_range in anomaly_free:
    for i in range(0, len(timeserie), day_size):
        day_serie = timeserie[i:i+day_size].squeeze(-1)
        date = date_range
        cleaned_dataset.append((day_serie, date))

# remove stride-caused duplicates, keeping imputed days with lower window variance (intuitively less anomalous)
df = pd.DataFrame(cleaned_dataset, columns=["timeserie", "date"])
df["var"] = df["timeserie"].apply(lambda x: x.var())
df["first_date"] = df["date"].apply(lambda x: x[0])
df = df.sort_values(by='var').groupby('first_date').first().reset_index()
cleaned_dataset = df[["timeserie", "date"]].values.tolist()
    
# merge consecutive days into a single timeserie
continuous_serie = []
for timeserie, date in cleaned_dataset:
    for j in range(len(timeserie)):
        continuous_serie.append((timeserie[j].item(), date[j]))

# save cleaned data
cleaned_load_serie = pd.DataFrame(continuous_serie, columns=["timeserie", "date"])
cleaned_load_serie.rename(columns={'timeserie': 'TOTALDEMAND'}, inplace=True)
cleaned_load_serie.to_csv(f"dataset/processed/{data_folder}/load_cleaned.csv", columns=["date", 'TOTALDEMAND'], index=False)

print(f"saved cleaned load serie to dataset/processed/{data_folder}/load_cleaned.csv")


# ---
# prepare data for forecasting    
# ---

from src.data.prepare_data_LF import run as prepare_data_LF_run
from src.data.prepare_data_LF import parse_args as prepare_data_LF_parse_args

default_prepare_data_LF_args = prepare_data_LF_parse_args()
default_prepare_data_LF_args.n_days = forecast_window_size
default_prepare_data_LF_args.raw_test_data_csv = f"dataset/processed/{data_folder}/load_clean_lf_test.csv"
default_prepare_data_LF_args.trg_test_save_data = f"dataset/processed/{data_folder}/lf_test_clean"

# cleaned data
default_prepare_data_LF_args.raw_train_data_csv = f"dataset/processed/{data_folder}/load_cleaned.csv"
default_prepare_data_LF_args.trg_train_save_data = f"dataset/processed/{data_folder}/lf_cleaned"
prepare_data_LF_run(default_prepare_data_LF_args)

# contamined data
default_prepare_data_LF_args.raw_train_data_csv = f"dataset/processed/{data_folder}/load_contam.csv"
default_prepare_data_LF_args.trg_train_save_data = f"dataset/processed/{data_folder}/lf_contam"
prepare_data_LF_run(default_prepare_data_LF_args)

