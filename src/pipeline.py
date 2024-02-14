import sys
sys.path.append("src/data")                 # data processing module
sys.path.append("src/anomaly_detection")    # AD module
sys.path.append("src/anomaly_imputation")   # AI module
sys.path.append("src/forecasting")          # LF module

import os
import shutil
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from utils.utils import set_seed


def run_pipeline(data_folder, 
                 exp_folder,
                 day_size, n_days, window_size, day_stride, day_contam_rate, data_contam_rate, flag_consec, outlier_threshold,  # anomaly detection parameters
                 forecast_model, forecast_window_size, forecast_day_stride, forecast_sequence_split,                            # forecasting parameters
                 save_figs):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)
    
    # ---
    # directories for results/plots/weights saving
    # ---

    results_folder = "results_0.4_0.05"
    # logs
    log_file_path = f"{results_folder}/{data_folder}/{exp_folder}/log.txt"
    # data
    raw_data_path = f"dataset/raw/{data_folder}"
    processed_data_path = f"dataset/processed/{data_folder}/{exp_folder}"
    AD_train_data_path = f"dataset/processed/{data_folder}/{exp_folder}/ad_train_contam"
    AD_test_data_path = f"dataset/processed/{data_folder}/{exp_folder}/ad_test_contam"
    imputation_data_path = f"dataset/processed/{data_folder}/{exp_folder}/ai_train/data"
    contam_load_csv_path = f"dataset/processed/{data_folder}/{exp_folder}/load_contam.csv"
    cleaned_load_csv_path = f"dataset/processed/{data_folder}/{exp_folder}/load_cleaned.csv"
    test_load_csv_path = f"dataset/processed/{data_folder}/{exp_folder}/load_clean_lf_test.csv"
    forecasting_clean_data = f"dataset/processed/{data_folder}/{exp_folder}/lf_cleaned"
    forecasting_contam_data = f"dataset/processed/{data_folder}/{exp_folder}/lf_contam"
    forecasting_test_data = f"dataset/processed/{data_folder}/{exp_folder}/lf_test_clean"
    # plots
    heatmaps_path = f"{results_folder}/{data_folder}/{exp_folder}/heatmaps"
    imputation_eval_plots_path = f"{results_folder}/{data_folder}/{exp_folder}/ai_eval_plots"
    imputation_infer_plots_path = f"{results_folder}/{data_folder}/{exp_folder}/imputation"
    forecasting_plots_path = f"{results_folder}/{data_folder}/{exp_folder}/forecasting"
    # weights
    weights_path = f"{results_folder}/{data_folder}/{exp_folder}/weights"

    folder_path_list = [AD_train_data_path
                        ,AD_test_data_path
                        ,imputation_data_path
                        ,forecasting_clean_data
                        ,forecasting_contam_data
                        ,forecasting_test_data
                        ,heatmaps_path
                        ,imputation_eval_plots_path
                        ,imputation_infer_plots_path
                        ,forecasting_plots_path
                        ,weights_path]
    file_path_list = [log_file_path, contam_load_csv_path, cleaned_load_csv_path, test_load_csv_path]
    
    for path in folder_path_list:
        os.makedirs(path, exist_ok=True)
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            else:
                shutil.rmtree(item_path)
    
    for path in file_path_list:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            os.remove(path)

    # ---
    # Generate synthetic data
    # ---
        
    if "INPG" in data_folder:
        from data_processing.process_INPG import run as prepare_data_AD_run
        from data_processing.process_INPG import parse_args as prepare_data_AD_parse_args

    if "AEMO" in data_folder:
        from data_processing.process_aemo import run as prepare_data_AD_run
        from data_processing.process_aemo import parse_args as prepare_data_AD_parse_args

    if "Park" in data_folder:
        from data_processing.process_park import run as prepare_data_AD_run
        from data_processing.process_park import parse_args as prepare_data_AD_parse_args

    default_process_data_AD_args = prepare_data_AD_parse_args()
    default_process_data_AD_args.raw_data_root = raw_data_path
    default_process_data_AD_args.trg_save_data = processed_data_path
    default_process_data_AD_args.log_file = log_file_path
    default_process_data_AD_args.day_size = day_size
    default_process_data_AD_args.n_days = n_days
    default_process_data_AD_args.day_stride = day_stride
    if "INPG" not in data_folder:
        default_process_data_AD_args.day_contam_rate = day_contam_rate
        default_process_data_AD_args.data_contam_rate = data_contam_rate

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
    default_AD_args.train_data_path = [AD_train_data_path, AD_test_data_path]
    default_AD_args.test_data_path = [AD_train_data_path, AD_test_data_path]
    default_AD_args.nbr_timesteps = window_size
    default_AD_args.contam_ratio = day_contam_rate
    default_AD_args.results_file = log_file_path
    default_AD_args.model_save_path = weights_path
    default_AD_args.eval_plots_path = os.path.dirname(log_file_path)
    # default_AD_args.without_soft_weight = True # set to True in case of nan values left in data

    AD_run(default_AD_args)

    # ---
    # find and impute anomalies
    # ---

    # load continuous csv of contaminated data
    load_serie = pd.read_csv(contam_load_csv_path, index_col=0, parse_dates=True)

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

    display_mask = True
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
                                                flag_highest_patch=False,
                                                flag_consec=flag_consec,  # False for INPG dataset, True otherwise
                                                flag_outliers=True,
                                                extend_to_patch=True,
                                                outlier_threshold=outlier_threshold,
                                                anom_idx_only=True)
                    
                    masked_data = timeserie.clone()
                    masked_data[anom_idx] = 0

                    mask = torch.ones_like(masked_data)
                    mask[anom_idx] = 0 
                    anomalous.append((masked_data, mask, date_range))

                    if save_figs:
                        # save heatmap of anomaly score for each timeserie
                        heatmap = heatmaps[i].reshape(1, -1)
                        heatmap_data = np.repeat(heatmap, len(timeserie)//heatmap.shape[1], axis=1) 
                        _, ax1 = plt.subplots(figsize=(10, 6))
                        ax1.imshow(heatmap_data, cmap="OrRd", aspect='auto')
                        ax1.set_yticks([])
                        cbar = plt.colorbar(ax1.imshow(heatmap_data, cmap="OrRd", aspect='auto'))
                        cbar.set_label('Anomaly Score', rotation=270, labelpad=0)
                        cbar.set_ticks([heatmap.min(), heatmap.max()])
                        cbar.set_ticklabels(['Low', 'High'])
                        ax2 = ax1.twinx()
                        ax2.plot(timeserie, label='Time Series', color='blue')
                        if display_mask:
                            scale = torch.max(timeserie) if torch.abs(torch.max(timeserie)) >= torch.abs(torch.min(timeserie)) else torch.min(timeserie)
                            mask *= scale # for better visualization, masked part is set to 0, rest is scaled to either max or min of the timeserie
                            ax2.plot(mask, label='Mask', color='green')
                        ax2.set_xlabel('Time')
                        ax2.set_ylabel('Value', color='blue')
                        ax2.tick_params('y', colors='blue')
                        ax2.yaxis.set_label_position('left')
                        ax2.yaxis.set_ticks_position('left')
                        plt.title('Time Series with Anomaly Score Heatmap')
                        plt.legend()
                        plt.savefig(f"{heatmaps_path}/{i}.png")
                        plt.close()
                i += 1

    # save anomaly free samples to train Imputation model
    for timeserie, date_range in anomaly_free:
        date = str(date_range[0].date()) + "_" + str(date_range[-1].date())
        dates = ''
        for d in range(n_days):
            dates += str(date_range[d*day_size].date()) + "_"
        np.save(os.path.join(imputation_data_path, str(dates)+'.npy'), timeserie.squeeze(-1))
    print(f"saved impuation training data to {imputation_data_path}.")

    # train anomaly imputation model on anomaly free samples
    from anomaly_imputation.main import parse_args as AI_parse_args
    from anomaly_imputation.main import train as AI_train
    from anomaly_imputation.autoencoder import LSTM_AE

    default_AI_args = AI_parse_args()
    default_AI_args.seq_len = window_size
    default_AI_args.mask_size = window_size // heatmaps[0].shape[0] # TODO make it a parameter
    default_AI_args.dataset_root = imputation_data_path
    default_AI_args.checkpoint_path = f"{weights_path}/checkpoint_ai.pt"
    default_AI_args.save_folder = imputation_eval_plots_path

    AI_train(default_AI_args)

    # infer anomaly imputation model on samples flagged as anomalous
    loaded_model = LSTM_AE(default_AI_args.seq_len, default_AI_args.no_features, default_AI_args.embedding_dim, default_AI_args.learning_rate, default_AI_args.every_epoch_print, default_AI_args.epochs, default_AI_args.patience, default_AI_args.max_grad_norm, default_AI_args.checkpoint_path, default_AI_args.seed)
    loaded_model.load()

    cleaned_anomalies = []
    for i, (masked_data, mask, date_range) in enumerate(anomalous):
        filled_ts = loaded_model.impute(masked_data.unsqueeze(0), mask)
        cleaned_anomalies.append((filled_ts, date_range))

        if save_figs:
            # save imputation visualization
            masked_data = masked_data.squeeze(0).squeeze(-1)
            mask = mask.squeeze(0).squeeze(-1)
            mask *= max(masked_data)/max(mask)
            plt.plot(masked_data, label="serie with missing values")
            plt.plot(mask, label="mask")
            plt.plot(filled_ts, label="serie with filled values")
            plt.legend()
            plt.savefig(os.path.join(imputation_infer_plots_path, f"{i}.png"))
            plt.clf()

    # ---
    # save cleaned (imputed) data for forecasting task
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

    # save cleaned data as continuous csv
    cleaned_load_serie = pd.DataFrame(continuous_serie, columns=["timeserie", "date"])
    feat_name = pd.read_csv(test_load_csv_path, index_col=0).columns[0]
    cleaned_load_serie.rename(columns={'timeserie': feat_name}, inplace=True)
    cleaned_load_serie.to_csv(cleaned_load_csv_path, columns=["date", feat_name], index=False)

    print(f"saved cleaned load serie to {cleaned_load_csv_path}")

    # ---
    # prepare data for forecasting    
    # ---

    if "INPG" in data_folder:
        from data_processing.process_LF_INPG import run as prepare_data_LF_run
        from data_processing.process_LF_INPG import parse_args as prepare_data_LF_parse_args
    else:
        from data_processing.process_LF import run as prepare_data_LF_run
        from data_processing.process_LF import parse_args as prepare_data_LF_parse_args

    default_process_data_LF_args = prepare_data_LF_parse_args()
    default_process_data_LF_args.n_days = forecast_window_size
    default_process_data_LF_args.day_size = day_size
    default_process_data_LF_args.day_stride = forecast_day_stride
    default_process_data_LF_args.trg_feature_name = feat_name
    default_process_data_LF_args.raw_test_data_csv = test_load_csv_path
    default_process_data_LF_args.trg_test_save_data = forecasting_test_data
    default_process_data_LF_args.log_file = log_file_path

    # process cleaned data
    default_process_data_LF_args.raw_train_data_csv = cleaned_load_csv_path
    default_process_data_LF_args.trg_train_save_data = forecasting_clean_data
    prepare_data_LF_run(default_process_data_LF_args)

    # process contamined data
    default_process_data_LF_args.raw_train_data_csv = contam_load_csv_path
    default_process_data_LF_args.trg_train_save_data = forecasting_contam_data
    prepare_data_LF_run(default_process_data_LF_args)

    # ---
    # run forecasting model
    # ---

    # run forecasting model on cleaned data
    from forecasting.main import run as LF_run
    from forecasting.main import parse_args as LF_parse_args

    default_LF_args = LF_parse_args()
    default_LF_args.timesteps = day_size * forecast_window_size
    default_LF_args.sequence_split = forecast_sequence_split
    default_LF_args.results_file = log_file_path

    # run forecasting model on cleaned data
    default_LF_args.train_dataset_path = forecasting_clean_data
    default_LF_args.test_dataset_path = forecasting_test_data
    # seq2seq model
    if forecast_model == "seq2seq" or forecast_model == "all":
        default_LF_args.model_choice = "seq2seq"
        default_LF_args.save_plots_path = f"{forecasting_plots_path}/seq2seq/cleaned"
        default_LF_args.checkpoint_path = f"{weights_path}/seq2seq/checkpoint_lf_clean.pt"
        smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = LF_run(default_LF_args)
        print(f"seq2seq: Cleaned data: smape={smape_loss:0.3f}, r2={r2_loss:0.3f}, mae={mae_loss:0.3f} ({mae_loss * (max_q_val - min_q_val):0.3f}), mse={mse_loss:0.3f} ({mse_loss * (max_q_val - min_q_val)**2:0.3f}), rmse={rmse_loss:0.3f} ({rmse_loss * (max_q_val - min_q_val):0.3f})", file=open(default_LF_args.results_file, "a"))
    # scinet model
    if forecast_model == "scinet" or forecast_model == "all":
        default_LF_args.model_choice = "scinet"
        default_LF_args.save_plots_path = f"{forecasting_plots_path}/scinet/cleaned"
        default_LF_args.checkpoint_path = f"{weights_path}/scinet/checkpoint_lf_clean.pt"
        smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = LF_run(default_LF_args)
        print(f"SCINet: Cleaned data: smape={smape_loss:0.3f}, r2={r2_loss:0.3f}, mae={mae_loss:0.3f} ({mae_loss * (max_q_val - min_q_val):0.3f}), mse={mse_loss:0.3f} ({mse_loss * (max_q_val - min_q_val)**2:0.3f}), rmse={rmse_loss:0.3f} ({rmse_loss * (max_q_val - min_q_val):0.3f})", file=open(default_LF_args.results_file, "a"))

    # run forecasting model on contamined data
    default_LF_args.train_dataset_path = forecasting_contam_data
    default_LF_args.test_dataset_path = forecasting_test_data
    # seq2seq model
    if forecast_model == "seq2seq" or forecast_model == "all":
        default_LF_args.model_choice = "seq2seq"
        default_LF_args.save_plots_path = f"{forecasting_plots_path}/seq2seq/contam"
        default_LF_args.checkpoint_path = f"{weights_path}/seq2seq/checkpoint_lf_contam.pt"
        smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = LF_run(default_LF_args)
        print(f"seq2seq: Cleaned data: smape={smape_loss:0.3f}, r2={r2_loss:0.3f}, mae={mae_loss:0.3f} ({mae_loss * (max_q_val - min_q_val):0.3f}), mse={mse_loss:0.3f} ({mse_loss * (max_q_val - min_q_val)**2:0.3f}), rmse={rmse_loss:0.3f} ({rmse_loss * (max_q_val - min_q_val):0.3f})", file=open(default_LF_args.results_file, "a"))
    # scinet model
    if forecast_model == "scinet" or forecast_model == "all":
        default_LF_args.model_choice = "scinet"
        default_LF_args.save_plots_path = f"{forecasting_plots_path}/scinet/contam"
        default_LF_args.checkpoint_path = f"{weights_path}/scinet/checkpoint_lf_contam.pt"
        smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = LF_run(default_LF_args)
        print(f"SCINet: Cleaned data: smape={smape_loss:0.3f}, r2={r2_loss:0.3f}, mae={mae_loss:0.3f} ({mae_loss * (max_q_val - min_q_val):0.3f}), mse={mse_loss:0.3f} ({mse_loss * (max_q_val - min_q_val)**2:0.3f}), rmse={rmse_loss:0.3f} ({rmse_loss * (max_q_val - min_q_val):0.3f})", file=open(default_LF_args.results_file, "a"))


if __name__ == "__main__":
    # parameters
    data_folder = "AEMO/SA"                                                 # dataset folder, must be in dataset/raw/
    exp_folder = "exp0"                                                     # folder for saving datasets, results, plots and weights, will be created in dataset/processed/ and in results/
    day_size = 24 if "INPG" in data_folder else 48                          # dataset resolution (samples per day)
    n_days = 1                                                              # window size for anomaly detection in terms of days
    window_size = day_size * n_days                                         # window size for anomaly detection in terms of data points
    day_stride = 1                                                          # sliding window stride for anomaly detection, a different stride may be set later for forecasting
    day_contam_rate = 0.02 if "INPG" in data_folder else 0.4                # day contamination ratio (% of days with anomalies, one anomaly per day)
    data_contam_rate = 0.05                                                 # data contamination ratio (% of data points with anomalies)
    flag_consec = "INPG" not in data_folder                                 # False for INPG dataset, True otherwise (anomaly type 1 and 2)
    outlier_threshold = 2.4 if "INPG" in data_folder else 2.5               # threshold for outlier detection
    forecast_window_size = 6                                                # window size for forecasting in days (including forecast day)
    forecast_day_stride = 1                                                 # stride for forecasting
    forecast_sequence_split = (forecast_window_size-1)/forecast_window_size # split ratio for forecasting (model input, forecast horizon)
    forecast_model = "all"                                                  # model to use for forecasting: "seq2seq", "scinet" or "all"
    save_figs = True                                                        # save visualization plots for anomaly detection and imputation

    # run pipeline (data processing, anomaly detection, anomaly imputation, forecasting with cleaned/contamined data)
    run_pipeline(data_folder, exp_folder, day_size, n_days, window_size, day_stride, day_contam_rate, data_contam_rate, flag_consec, outlier_threshold, forecast_model, forecast_window_size, forecast_day_stride, forecast_sequence_split, save_figs)
