import sys
sys.path.append("src/data")                 # data preparation module
sys.path.append("src/anomaly_detection")    # AD module
sys.path.append("src/anomaly_imputation")   # AI module
sys.path.append("src/forecasting")          # LF module

import os
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

from utils.utils import set_seed
from utils.utils import make_clean_folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0)

# parameters
data_folder = "INPG"          # dataset folder, must be in dataset/processed/
day_size = 24                                   # dataset resolution
n_days = 1                                      # window size for anomaly detection
window_size = day_size * n_days                 # window size for anomaly detection
day_stride = 1                                  # for anomaly detection, seperate stride for forecasting
contam_ratio = 0.1                              # contamination ratio for anomaly detection (% of days with anomalies, one anomaly per day)
forecast_window_size = 5                        # window size for forecasting
forecast_day_stride = 1                         # stride for forecasting
save_figs = True                                # save plots of anomaly detection and imputation
imp_trained = False                             # if True, skip training of anomaly imputation model

# prepare directories for results/plots/weights saving
save_imputation_train_path = f"dataset/processed/{data_folder}/ai_train/data"
save_heatmaps_path = f"results/{data_folder}/heatmaps"
save_imputation_path = f"results/{data_folder}/imputation"
save_forecasting_cleaned_path = f"results/{data_folder}/forecasting/cleaned"
save_forecasting_contam_path = f"results/{data_folder}/forecasting/contam"

path_list = [save_imputation_train_path, save_heatmaps_path, save_imputation_path, save_forecasting_cleaned_path, save_forecasting_contam_path]
for path in path_list:
    make_clean_folder(path)

# ---
# Generate synthetic data
# ---
# from data_processing.process_aemo import run as prepare_data_AD_run
# from data_processing.process_aemo import parse_args as prepare_data_AD_parse_args

# from data_processing.process_park import run as prepare_data_AD_run
# from data_processing.process_park import parse_args as prepare_data_AD_parse_args
    
from data_processing.process_INPG import run as prepare_data_AD_run
from data_processing.process_INPG import parse_args as prepare_data_AD_parse_args

default_process_data_AD_args = prepare_data_AD_parse_args()
default_process_data_AD_args.raw_data_root = f"dataset/raw/{data_folder}"
default_process_data_AD_args.trg_save_data = f"dataset/processed/{data_folder}"
default_process_data_AD_args.log_file = f"results/{data_folder}/log.txt"
default_process_data_AD_args.day_size = day_size
default_process_data_AD_args.n_days = n_days
default_process_data_AD_args.day_stride = day_stride

min_q_val, max_q_val = prepare_data_AD_run(default_process_data_AD_args)

# ---
# train AD model
# ---
from anomaly_detection.main import run as AD_run
from anomaly_detection.main import parse_args as AD_parse_args
from anomaly_detection.main import get_coreset
from anomaly_detection.common import FaissNN
from anomaly_detection.postprocessing import heatmap_postprocess

default_AD_args = AD_parse_args()
default_AD_args.train_data_path = [f"dataset/processed/{data_folder}/ad_train_contam", f"dataset/processed/{data_folder}/ad_test_contam"]
default_AD_args.test_data_path = [f"dataset/processed/{data_folder}/ad_train_contam", f"dataset/processed/{data_folder}/ad_test_contam"]
default_AD_args.nbr_timesteps = window_size
default_AD_args.contam_ratio = contam_ratio
default_AD_args.model_save_path = f"results/{data_folder}/weights"
# default_AD_args.without_soft_weight = True # set true in case of nan values in data

AD_run(default_AD_args)

# ---
# find and impute anomalies
# ---

# load continuous "load.csv" data
load_serie = pd.read_csv(f"dataset/processed/{data_folder}/load_contam.csv", index_col=0, parse_dates=True)

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
load_dataset = TensorDataset(torch.tensor(load_windows, dtype=torch.float))  # batch_size x seq_len x 1=feature_dim
infer_dataloader = DataLoader(load_dataset, batch_size=32, shuffle=False)    # GT not needed

# load AD model
coreset = get_coreset(default_AD_args, device)
coreset.load_from_path(default_AD_args.model_save_path, device, FaissNN(False, 4))

# infer AD model
scores, heatmaps, _, _, _ = coreset.predict(infer_dataloader)
scores = (scores - coreset.min_score) / (coreset.max_score - coreset.min_score + 1e-5)
scores = np.mean(scores, axis=0)

heatmaps = np.array(heatmaps)
heatmaps = (heatmaps - coreset.min_heatmap_scores) / (coreset.max_heatmap_scores - coreset.min_heatmap_scores)
heatmaps = np.mean(heatmaps, axis=-1)

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

            else:
                anom_idx = heatmap_postprocess(timeserie, heatmap, 
                                               flag_highest_patch=True, extend_to_patch=True, 
                                               anom_idx_only=True)
                
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
                    plt.savefig(f"{save_heatmaps_path}/{i}.png")
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
from anomaly_imputation.main import parse_args as AI_parse_args
from anomaly_imputation.main import train as AI_train
from anomaly_imputation.autoencoder import LSTM_AE

default_AI_args = AI_parse_args()
default_AI_args.seq_len = window_size
default_AI_args.mask_size = window_size // heatmaps[0].shape[0] 
default_AI_args.dataset_root = save_imputation_train_path
default_AI_args.checkpoint_path = f"results/{data_folder}/weights/checkpoint_ai.pt"
default_AI_args.save_folder = f"results/{data_folder}/ai_eval_plots"

if not imp_trained:
    AI_train(default_AI_args)

# infer anomaly detection and impute anomalies on anomalous samples
loaded_model = LSTM_AE(default_AI_args.seq_len, default_AI_args.no_features, default_AI_args.embedding_dim, default_AI_args.learning_rate, default_AI_args.every_epoch_print, default_AI_args.epochs, default_AI_args.patience, default_AI_args.max_grad_norm, default_AI_args.checkpoint_path, default_AI_args.seed)
loaded_model.load()

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
feat_name = pd.read_csv(f"dataset/processed/{data_folder}/load_clean_lf_test.csv", index_col=0).columns[0]
cleaned_load_serie.rename(columns={'timeserie': feat_name}, inplace=True)
cleaned_load_serie.to_csv(f"dataset/processed/{data_folder}/load_cleaned.csv", columns=["date", feat_name], index=False)

print(f"saved cleaned load serie to dataset/processed/{data_folder}/load_cleaned.csv")


# ---
# prepare data for forecasting    
# ---

# from data_processing.process_LF import run as prepare_data_LF_run
# from data_processing.process_LF import parse_args as prepare_data_LF_parse_args

from data_processing.process_LF_INPG import run as prepare_data_LF_run
from data_processing.process_LF_INPG import parse_args as prepare_data_LF_parse_args

default_process_data_LF_args = prepare_data_LF_parse_args()
default_process_data_LF_args.n_days = forecast_window_size
default_process_data_LF_args.day_size = day_size
default_process_data_LF_args.day_stride = forecast_day_stride
default_process_data_LF_args.trg_feature_name = feat_name
default_process_data_LF_args.raw_test_data_csv = f"dataset/processed/{data_folder}/load_clean_lf_test.csv"
default_process_data_LF_args.trg_test_save_data = f"dataset/processed/{data_folder}/lf_test_clean"
default_process_data_LF_args.log_file = f"results/{data_folder}/log.txt"

# cleaned data
default_process_data_LF_args.raw_train_data_csv = f"dataset/processed/{data_folder}/load_cleaned.csv"
default_process_data_LF_args.trg_train_save_data = f"dataset/processed/{data_folder}/lf_cleaned"
prepare_data_LF_run(default_process_data_LF_args)

# contamined data
default_process_data_LF_args.raw_train_data_csv = f"dataset/processed/{data_folder}/load_contam.csv"
default_process_data_LF_args.trg_train_save_data = f"dataset/processed/{data_folder}/lf_contam"
prepare_data_LF_run(default_process_data_LF_args)


# ---
# run forecasting model
# ---

# run forecasting model on cleaned data
from forecasting.main import run as LF_run
from forecasting.main import parse_args as LF_parse_args

default_LF_args = LF_parse_args()
default_LF_args.timesteps = day_size * forecast_window_size
default_LF_args.sequence_split = (forecast_window_size-1)/forecast_window_size
default_LF_args.results_file = f"results/{data_folder}/log.txt"

# run forecasting model on cleaned data
default_LF_args.train_dataset_path = f"dataset/processed/{data_folder}/lf_cleaned"
default_LF_args.test_dataset_path = f"dataset/processed/{data_folder}/lf_test_clean"
default_LF_args.save_plots_path = f"results/{data_folder}/forecasting/cleaned"
default_LF_args.checkpoint_path = f"results/{data_folder}/weights/checkpoint_lf_clean.pt"

smape_loss, mae_loss, mse_loss, rmse_loss, mape_loss, mase_loss, r2_loss = LF_run(default_LF_args)
print(f"Cleaned data (real scale): smape={smape_loss}, mae={mae_loss * (max_q_val - min_q_val)}, mse={mse_loss * (max_q_val - min_q_val)**2}, rmse={rmse_loss * (max_q_val - min_q_val)}, mape={mape_loss}, mase={mase_loss}, r2={r2_loss}", file=open(default_LF_args.results_file, "a"))

# run forecasting model on contamined data
default_LF_args.train_dataset_path = f"dataset/processed/{data_folder}/lf_contam"
default_LF_args.test_dataset_path = f"dataset/processed/{data_folder}/lf_test_clean"
default_LF_args.save_plots_path = f"results/{data_folder}/forecasting/contam"
default_LF_args.checkpoint_path = f"results/{data_folder}/weights/checkpoint_lf_contam.pt"

smape_loss, mae_loss, mse_loss, rmse_loss, mape_loss, mase_loss, r2_loss = LF_run(default_LF_args)
print(f"Contamined data (real scale): smape={smape_loss}, mae={mae_loss * (max_q_val - min_q_val)}, mse={mse_loss * (max_q_val - min_q_val)**2}, rmse={rmse_loss * (max_q_val - min_q_val)}, mape={mape_loss}, mase={mase_loss}, r2={r2_loss}", file=open(default_LF_args.results_file, "a"))

# sMAPE is large for the INPG dataset, because the load is sometimes very low (~0), other metrics are more relevant in this case.
