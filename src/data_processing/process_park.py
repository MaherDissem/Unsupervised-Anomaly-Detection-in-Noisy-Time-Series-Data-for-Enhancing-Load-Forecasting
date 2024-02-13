import os
import argparse
import glob
import numpy as np
import pandas as pd

import sys
sys.path.append("./src")

from utils.utils import set_seed
from data_processing.synth_anomaly import SynthLoadAnomaly
from data_processing.fill_missing_values import fill_missing_values


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for anomaly detection model training and evaluation.")
    parser.add_argument("--raw_data_root",        type=str,   default="dataset/raw/Park/Commercial/30_minutes", help="Path to raw data root")
    parser.add_argument("--trg_save_data",        type=str,   default="dataset/processed/Park/Commercial/30_minutes", help="Path to save processed data")
    parser.add_argument("--load_feature_name",    type=str,   default="Power (kW)", help="Name of the load feature")
    parser.add_argument("--date_feature_name",    type=str,   default="Time", help="Name of the date_time feature")
    parser.add_argument("--day_size",             type=int,   default=48, help="Size of a day")
    parser.add_argument("--n_days",               type=int,   default=1, help="Number of days")
    parser.add_argument("--day_stride",           type=int,   default=1, help="Day stride for sliding window")
    parser.add_argument("--day_contam_rate",      type=float, default=0.4, help="Percentage of days with anomalies")
    parser.add_argument("--data_contam_rate",     type=float, default=0.05, help="Percentage of datapoints with anomalies") # these two parameters are used together to determine the spread and average length of anomalies type 1 and 2
    parser.add_argument("--contam_clean_ratio",   type=float, default=0.7, help="Clean data save ratio (forcasting model is later evaluated on this clean data)")
    parser.add_argument("--ad_split_ratio",       type=float, default=0.7, help="Anomaly detection train-test split ratio")
    parser.add_argument("--seed",                 type=int,   default=0, help="Random seed")
    parser.add_argument("--log_file",             type=str,   default="results/results.txt", help="Path of file to log to") # quantiles must be saved to later scale back metrics
    args = parser.parse_args()
    return args


def run(args):
    set_seed(args.seed)
    load = pd.DataFrame()
    sampling_rate = os.path.basename(args.raw_data_root)
    building_type = os.path.basename(os.path.dirname(args.raw_data_root))
    data_root = os.path.dirname(os.path.dirname(args.raw_data_root))

    csv_paths = glob.iglob(os.path.join(data_root, '*', sampling_rate, f"*{building_type}", "*.xlsx"))

    for csv_path in csv_paths:
        try: 
            csv_file = pd.read_excel(csv_path)
            csv_file = csv_file[[args.date_feature_name, args.load_feature_name]]
            csv_file[args.date_feature_name] = pd.to_datetime(csv_file[args.date_feature_name], format="%Y-%m-%d %H:%M:%S")
            na_perc = csv_file[args.load_feature_name].isna().sum()/len(csv_file[args.load_feature_name])
            zeros_perc = (csv_file[args.load_feature_name] == 0).sum()/len(csv_file[args.load_feature_name])
            if na_perc > 0 or zeros_perc>0.05: continue
        except Exception as e:
            print(e)
        load = pd.concat([load, csv_file], axis=0)
    
    if len(load) == 0: raise Exception("No data found!")

    load.set_index(args.date_feature_name, inplace=True)
    load.sort_index(inplace=True)
    
    # remove duplicate indices
    load = load[~load.index.duplicated()]

    # fill missing values
    idx = pd.date_range(load.index[0], load.index[-1], freq="30T") # TODO: make frequency dynamic
    load = load.reindex(idx, fill_value=np.nan)
    load = fill_missing_values(load, args.day_size)
    
    # split contam data into train and test sets for anomaly detection model
    N = int(args.contam_clean_ratio*len(load))//args.day_size*args.day_size

    contaminated_load = load[:N]
    clean_load = load[N:]
    ad_load = contaminated_load.copy()

    # contaminate data with synthetic anomalies    
    anomaly_generator = SynthLoadAnomaly()
    print(f"Anomaly generator probabilities: {anomaly_generator.prob_1}, {anomaly_generator.prob_2}, {anomaly_generator.prob_3}, {anomaly_generator.prob_4}, {anomaly_generator.prob_softstart}, {anomaly_generator.prob_extreme}", file=open(args.log_file, "a"))
    
    def contam_load(load, anomaly_generator, day_contam_rate, data_contam_rate, feature_name, day_size, seed=0):
        """contaminate load dataframe with synthetic anomalies"""

        np.random.seed(seed)
        anomaly_generator.__init__(seed=seed) # set seed and reset probabilties (modified later in the function)

        gt = np.zeros(len(load))
        n_days = len(load)//day_size
        n_contam_days = int(day_contam_rate*len(load)//day_size)
        contam_days = np.random.choice(range(0, len(load)//day_size), n_contam_days, replace=False)
        
        cur_contam = 0
        cur_contam_days = 0
        trg_contam = int(data_contam_rate*len(load))
        
        # calculate average anomaly length for type 1 and type 2 anomalies to achieve the target contamination rate
        avg_anom_1_length = (trg_contam - n_contam_days*anomaly_generator.prob_3 - n_contam_days*anomaly_generator.prob_4) / ((anomaly_generator.prob_1 + anomaly_generator.prob_2) / anomaly_generator.prob_1) / (n_contam_days * anomaly_generator.prob_1)
        avg_anom_2_length = (trg_contam - n_contam_days*anomaly_generator.prob_3 - n_contam_days*anomaly_generator.prob_4) / ((anomaly_generator.prob_1 + anomaly_generator.prob_2) / anomaly_generator.prob_2) / (n_contam_days * anomaly_generator.prob_2)
        anom1_len_var = avg_anom_1_length/2
        anom2_len_var = avg_anom_2_length/2
        print(f"avg_anom_1_length: {avg_anom_1_length}, avg_anom_2_length: {avg_anom_2_length}", file=open(args.log_file, "a"))
        
        for day in range(n_days):
            day_st = day*day_size
            day_end = day_st + day_size
            seq_gt = gt[day_st: day_end]

            if day in contam_days:
                if cur_contam_days >= n_contam_days*0.9:
                    # for the last chunk of data, we contaminate with the exact number of anomalies needed to reach the target contamination rate
                    avg_anom_1_length = (trg_contam - cur_contam) / ((anomaly_generator.prob_1 + anomaly_generator.prob_2) / anomaly_generator.prob_1) / ((n_contam_days - cur_contam_days) * anomaly_generator.prob_1)
                    avg_anom_2_length = (trg_contam - cur_contam) / ((anomaly_generator.prob_1 + anomaly_generator.prob_2) / anomaly_generator.prob_2) / ((n_contam_days - cur_contam_days) * anomaly_generator.prob_2)
                    anom1_len_var = 0
                    anom2_len_var = 0
                    anomaly_generator.prob_1 += anomaly_generator.prob_3
                    anomaly_generator.prob_2 += anomaly_generator.prob_4
                    anomaly_generator.prob_3 = 0
                    anomaly_generator.prob_4 = 0
                
                # contaminate day randomly (anomaly probabilities are given to the generator)
                sequence = load[feature_name].values[day_st: day_end]
                anomalous_sequence, anom_idx = anomaly_generator.inject_anomaly(sequence, 1,
                                                                                avg_anom_1_length, anom1_len_var,
                                                                                avg_anom_2_length, anom2_len_var)
                load[feature_name].values[day_st: day_end] = anomalous_sequence

                # update gt
                for anom_id in anom_idx: 
                    if not seq_gt[anom_id]: # handles adding anomalies to contaminated load (not implemented)
                        seq_gt[anom_id] = 1 
                        cur_contam += 1
                cur_contam_days += 1

            gt[day_st: day_end] = seq_gt

            if cur_contam >= trg_contam: 
                break

        return load, gt

    def extract_consec_days(load, gt_load, day0, n_days, day_size):
        """return n_days consecutive days starting at day0 from load dataframe"""

        sequence, gt = [], []
        start = day0
        end = start + day_size

        for day in range(n_days):
            sequence.extend(load[args.load_feature_name].values[start: end])
            gt.extend(gt_load[start: end])
            start += day_size
            end += day_size
        return np.array(sequence), np.array(gt)

    def build_dataset(load, n_days, day_size, day_stride, contam_data=True):
        """
            build a dataset from load dataframe using a sliding window of size n_days and stride of 1 day 
            while contamining the data with synthetic anomalies
        """
        if contam_data:
            load, gt_load = contam_load(load, anomaly_generator, args.day_contam_rate, args.data_contam_rate, args.load_feature_name, day_size, args.seed)
        else:
            gt_load = [[0]*day_size]*(len(load)//day_size)
        
        time_wind = []
        gt_time_wind = []
        datetime_wind = []

        day_idx = 0
        while day_idx < len(load)//day_size - n_days:
            day0 = day_idx*day_size
            sequence, gt = extract_consec_days(load, gt_load, day0, n_days, day_size)

            time_wind.append(sequence)
            gt_time_wind.append(gt)
            first_date = str(load.index[day0]).replace(':', '')
            last_date = str(load.index[day0 + n_days*day_size-1]).replace(':', '')
            datetime_wind.append(f"{first_date} - {last_date}")
            day_idx += day_stride

        return time_wind, gt_time_wind, datetime_wind

    ts_windows, gt_windows, date_windows = build_dataset(ad_load, args.n_days, args.day_size, args.day_stride, contam_data=True)
    
    M = int(args.ad_split_ratio*len(ts_windows))
    ad_train_windows, ad_test_windows = ts_windows[:M], ts_windows[M:]
    gt_ad_train_windows, gt_ad_test_windows = gt_windows[:M], gt_windows[M:]
    date_ad_train_windows, date_ad_test_windows = date_windows[:M], date_windows[M:]

    datapoint_contam_ratio = np.array(gt_ad_train_windows+gt_ad_test_windows).sum() / (len(gt_ad_train_windows+gt_ad_test_windows)*args.day_size)

    # normalize data
    min_quantile = 0.01
    max_quantile = 0.99

    min_q_val = clean_load.quantile(min_quantile).item()
    max_q_val = clean_load.quantile(max_quantile).item()

    def scale_windows(windows_list, min_q_val, max_q_val):
        scaled_windows = []
        for window in windows_list:
            window = (window - min_q_val) / (max_q_val - min_q_val)
            scaled_windows.append(window)
        return scaled_windows

    ad_train_windows = scale_windows(ad_train_windows, min_q_val, max_q_val)
    ad_test_windows = scale_windows(ad_test_windows, min_q_val, max_q_val)

    # save data
    # remove existing files in save target root folder
    existing_files = glob.glob(os.path.join(args.trg_save_data, "*", "*", "*.npy"))
    for f in existing_files:
        os.remove(f)

    # create save target folders if they don't exist
    os.makedirs(os.path.join(args.trg_save_data, "lf_test_clean", "data"), exist_ok=True)
    os.makedirs(os.path.join(args.trg_save_data, "ad_train_contam", "data"), exist_ok=True)
    os.makedirs(os.path.join(args.trg_save_data, "ad_train_contam", "gt"), exist_ok=True)
    os.makedirs(os.path.join(args.trg_save_data, "ad_test_contam", "data"), exist_ok=True)
    os.makedirs(os.path.join(args.trg_save_data, "ad_test_contam", "gt"), exist_ok=True)

    # save contam ad train data
    for i, (sample, sample_gt, sample_date) in enumerate(zip(ad_train_windows, gt_ad_train_windows, date_ad_train_windows)):
        if np.isnan(sample).any(): continue
        np.save(os.path.join(args.trg_save_data, "ad_train_contam", "data", sample_date), sample)
        np.save(os.path.join(args.trg_save_data, "ad_train_contam", "gt", sample_date), sample_gt)

    # save contam ad test data
    for i, (sample, sample_gt, sample_date) in enumerate(zip(ad_test_windows, gt_ad_test_windows, date_ad_test_windows)):
        if np.isnan(sample).any(): continue
        np.save(os.path.join(args.trg_save_data, "ad_test_contam", "data", sample_date), sample)
        np.save(os.path.join(args.trg_save_data, "ad_test_contam", "gt", sample_date), sample_gt)

    # log results
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    print(args, file=open(args.log_file, "a"))
    print(f"Number of ad_train_contam windows: {len(ad_train_windows)}", file=open(args.log_file, "a"))
    print(f"Number of ad_test_contam windows: {len(ad_test_windows)}", file=open(args.log_file, "a"))

    print(f"{args.day_contam_rate*100:.2f}% of days are contaminated.", file=open(args.log_file, "a"))
    print(f"{datapoint_contam_ratio*100:.2f}% of datapoints are contaminated.", file=open(args.log_file, "a"))

    print(f"min_quantile={min_quantile:0.3f} -> value={min_q_val}", file=open(args.log_file, "a"))
    print(f"max_quantile={max_quantile:0.3f} -> value={max_q_val}", file=open(args.log_file, "a"))

    # save clean load for forecasting model evaluation
    clean_load = (clean_load - min_q_val) / (max_q_val - min_q_val)
    clean_load.rename_axis("date", inplace=True)
    clean_load.to_csv(os.path.join(args.trg_save_data, "load_clean_lf_test.csv"))

    # save contaminated load serie to infer AD/AI models after training
    contam_full_load, gt_full_load = contam_load(contaminated_load, anomaly_generator, args.day_contam_rate, args.data_contam_rate, args.load_feature_name, args.day_size, args.seed+1) # for a more realistic scenario, data contamination here is different than for the AD model's training. AD is unsupervised anyway.
    scaled_load = (contam_full_load - min_q_val) / (max_q_val - min_q_val)
    scaled_load.rename_axis("date", inplace=True)
    scaled_load.to_csv(os.path.join(args.trg_save_data, "load_contam.csv"))
    gt_full_load = pd.Series(gt_full_load, index=scaled_load.index[:len(gt_full_load)]).rename("gt", inplace=True)
    pd.Series(gt_full_load).to_csv(os.path.join(args.trg_save_data, "load_contam_gt.csv"))
    print('Dataset ready!')

    return min_q_val, max_q_val


if __name__ == "__main__":
    args = parse_args()
    run(args)


# Dataset saved to dataset/processed/Park\Commercial\30_minutes
# percentage of nan values: 3.89%
# Dataset saved to dataset/processed/Park\Office\30_minutes
# percentage of nan values: 2.66%
# Dataset saved to dataset/processed/Park\Residential\30_minutes
# percentage of nan values: 8.56%
# Dataset saved to dataset/processed/Park\Public\30_minutes
# percentage of nan values: 31.96%
            
# Dataset saved to dataset/processed/Park\Commercial\1_hour
# percentage of nan values: 3.97%
# Dataset saved to dataset/processed/Park\Office\1_hour
# percentage of nan values: 2.64%
# Dataset saved to dataset/processed/Park\Residential\1_hour
# percentage of nan values: 8.57%
# Dataset saved to dataset/processed/Park\Public\1_hour
# percentage of nan values: 31.90%

