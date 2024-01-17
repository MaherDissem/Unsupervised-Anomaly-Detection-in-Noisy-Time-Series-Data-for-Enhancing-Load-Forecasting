import argparse
import os
import glob
import numpy as np
import pandas as pd

import sys
sys.path.append("./src")

from utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for anomaly detection model training and evaluation.")
    parser.add_argument("--raw_data_root",        type=str,   default="dataset/raw/Yahoo/A2Benchmark", help="Path to raw data root")
    parser.add_argument("--trg_save_data",        type=str,   default="dataset/processed/Yahoo/A2Benchmark", help="Path to save processed data")
    parser.add_argument("--feat_feature_name",    type=str,   default="value", help="Name of the feat feature")
    parser.add_argument("--date_feature_name",    type=str,   default="timestamp", help="Name of the date_time feature")
    parser.add_argument("--gt_name",              type=str,   default="is_anomaly", help="Name of the gt feature")
    parser.add_argument("--day_size",             type=int,   default=24, help="Size of a day")
    parser.add_argument("--n_days",               type=int,   default=1, help="Number of days")
    parser.add_argument("--day_stride",           type=int,   default=1, help="Day stride for sliding window")
    parser.add_argument("--ad_split_ratio",       type=float, default=0.7, help="Anomaly detection train-test split ratio")
    parser.add_argument("--seed",                 type=int,   default=0, help="Random seed")
    parser.add_argument("--log_file",             type=str,   default="results/results.txt", help="Path of file to log to") # quantiles must be saved to later scale back metrics
    args = parser.parse_args()
    return args


def run(args):
    set_seed(args.seed)
    data = pd.DataFrame()
    csv_paths = glob.iglob(os.path.join(args.raw_data_root, "*.csv"))

    for csv_path in csv_paths:
        try: 
            csv_file = pd.read_csv(csv_path)
            csv_file = csv_file[[args.date_feature_name, args.feat_feature_name, args.gt_name]]
        except Exception as e:
            print(e)
            continue
        data = pd.concat([data, csv_file], axis=0)


    # split data into train and test sets for anomaly detection model
    M = int(args.ad_split_ratio*(len(data)))//args.day_size*args.day_size
    ad_train_feat = data[:M]
    ad_test_feat = data[M:]

    def extract_consec_days(feat, gt_feat, day0, n_days, day_size):
        """return n_days consecutive days starting at day0 from feat dataframe"""

        sequence, gt = [], []
        start = day0
        end = start + day_size

        for day in range(n_days):
            sequence.extend(feat[start: end])
            gt.extend(gt_feat[start: end])
            start += day_size
            end += day_size
        return np.array(sequence), np.array(gt)

    def build_dataset(data, n_days, day_size, day_stride):
        """
            build a dataset from feat dataframe using a sliding window of size n_days and stride of 1 day 
            while contamining the data with synthetic anomalies
        """        
        time_wind = []
        gt_time_wind = []
        feat = data[args.feat_feature_name].values
        gt_feat = data[args.gt_name].values

        day_idx = 0
        while day_idx < len(feat)//day_size - n_days:
            day0 = day_idx*day_size
            sequence, gt = extract_consec_days(feat, gt_feat, day0, n_days, day_size)

            time_wind.append(sequence)
            gt_time_wind.append(gt)
            day_idx += day_stride

        return time_wind, gt_time_wind


    ad_train_windows, gt_ad_train_windows = build_dataset(ad_train_feat, args.n_days, args.day_size, args.day_stride)
    ad_test_windows, gt_ad_test_windows = build_dataset(ad_test_feat, args.n_days, args.day_size, args.day_stride)

    # normalize data
    min_quantile = 0.01
    max_quantile = 0.99

    min_q_val = data[args.feat_feature_name].quantile(min_quantile).item()
    max_q_val = data[args.feat_feature_name].quantile(max_quantile).item()

    def scale_windows(windows_list, min_q_val, max_q_val):
        scaled_windows = []
        for window in windows_list:
            min_q_val = window.min()
            max_q_val = window.max()
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

    # crete save target folders if they don't exist
    os.makedirs(os.path.join(args.trg_save_data, "ad_train_contam", "data"), exist_ok=True)
    os.makedirs(os.path.join(args.trg_save_data, "ad_train_contam", "gt"), exist_ok=True)
    os.makedirs(os.path.join(args.trg_save_data, "ad_test_contam", "data"), exist_ok=True)
    os.makedirs(os.path.join(args.trg_save_data, "ad_test_contam", "gt"), exist_ok=True)

    # save ad train data
    for i, (sample, sample_gt) in enumerate(zip(ad_train_windows, gt_ad_train_windows)):
        if np.isnan(sample).any(): continue
        np.save(os.path.join(args.trg_save_data, "ad_train_contam", "data", f"{i}.npy"), sample)
        np.save(os.path.join(args.trg_save_data, "ad_train_contam", "gt", f"{i}.npy"), sample_gt)

    # save ad test data
    for i, (sample, sample_gt) in enumerate(zip(ad_test_windows, gt_ad_test_windows)):
        if np.isnan(sample).any(): continue
        np.save(os.path.join(args.trg_save_data, "ad_test_contam", "data", f"{i}.npy"), sample)
        np.save(os.path.join(args.trg_save_data, "ad_test_contam", "gt", f"{i}.npy"), sample_gt)

    # log results
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    print(args, file=open(args.log_file, "a"))
    print(f"Number of ad_train_contam windows: {len(ad_train_windows)}", file=open(args.log_file, "a"))
    print(f"Number of ad_test_contam windows: {len(ad_test_windows)}", file=open(args.log_file, "a"))

    print(f"min_quantile={min_quantile:0.3f} -> value={min_q_val}", file=open(args.log_file, "a"))
    print(f"max_quantile={max_quantile:0.3f} -> value={max_q_val}", file=open(args.log_file, "a"))
    

if __name__ == "__main__":
    args = parse_args()
    run(args)
    print("Done!")
