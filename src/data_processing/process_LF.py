import argparse
import os
import glob
import numpy as np
import pandas as pd

import sys
sys.path.append("./src")

from utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for forecasting model training and evaluation.")
    parser.add_argument("--raw_train_data_csv",   type=str,   default="dataset/processed/Park/Office/30_minutes/exp0/load_cleaned.csv", help="Path to raw data root")
    parser.add_argument("--trg_train_save_data",  type=str,   default="dataset/processed/Park/Office/30_minutes/exp0/lf_cleaned", help="Path to save processed data")
    
    parser.add_argument("--raw_test_data_csv",    type=str,   default="dataset/processed/Park/Office/30_minutes/exp0/load_clean_lf_test.csv", help="Path to raw data root")
    parser.add_argument("--trg_test_save_data",   type=str,   default="dataset/processed/Park/Office/30_minutes/exp0/lf_test_clean", help="Path to save processed data")
    
    parser.add_argument("--trg_feature_name",     type=str,   default="Power (kW)", help="Name of the feat feature")
    parser.add_argument("--date_feature_name",    type=str,   default="date", help="Name of the date_time feature")

    parser.add_argument("--day_size",             type=int,   default=48, help="Size of a day")
    parser.add_argument("--n_days",               type=int,   default=6, help="Number of days")
    parser.add_argument("--day_stride",           type=int,   default=1, help="Day stride for sliding window")

    parser.add_argument("--seed",                 type=int,   default=0, help="Random seed")
    parser.add_argument("--log_file",             type=str,   default="results/results.txt", help="Path of file to log to") # quantiles must be saved to later scale back metrics
    args = parser.parse_args()
    return args


def run(args):
    set_seed(args.seed)
    train_data = pd.read_csv(args.raw_train_data_csv)
    test_data = pd.read_csv(args.raw_test_data_csv)

    train_data[args.date_feature_name] = pd.to_datetime(train_data[args.date_feature_name], format="%Y-%m-%d %H:%M:%S")
    test_data[args.date_feature_name] = pd.to_datetime(test_data[args.date_feature_name], format="%Y-%m-%d %H:%M:%S")

    train_data.set_index(args.date_feature_name, inplace=True)
    test_data.set_index(args.date_feature_name, inplace=True)

    # create calendar features (cycle features)
    def cyclical_encoding(data: pd.Series, cycle_length: int) -> pd.DataFrame:
        """
        Encode a cyclical feature with two new features sine and cosine.
        The minimum value of the feature is assumed to be 0. The maximum value
        of the feature is passed as an argument.
        
        Parameters
        ----------
        data : pd.Series
            Series with the feature to encode.
        cycle_length : int
            The length of the cycle. For example, 12 for months, 24 for hours, etc.
            This value is used to calculate the angle of the sin and cos.

        Returns
        -------
        pd.DataFrame
            Dataframe with the two new features sin and cos.

        """

        sin = np.sin(2 * np.pi * data/cycle_length)
        cos = np.cos(2 * np.pi * data/cycle_length)
        result =  pd.DataFrame({
                    f"{data.name}_sin": sin,
                    f"{data.name}_cos": cos
                })

        return result
    
    # create cyclical features
    def get_calendar_cyclical_features(data):
        from feature_engine.datetime import DatetimeFeatures
        transformer = DatetimeFeatures(
                        variables           = "index",
                        features_to_extract = "all" # It is also possible to select specific features
                    )
        calendar_features = transformer.fit_transform(data)

        month_encoded = cyclical_encoding(calendar_features['month'], cycle_length=12)
        day_of_week_encoded = cyclical_encoding(calendar_features['day_of_week'], cycle_length=7)
        hour_encoded = cyclical_encoding(calendar_features['hour'], cycle_length=24)

        cyclical_features = pd.concat([month_encoded, day_of_week_encoded, hour_encoded], axis=1)
        return cyclical_features

    train_cyclical_features = get_calendar_cyclical_features(train_data)
    train_data = pd.concat([train_data, train_cyclical_features], axis=1)

    test_cyclical_features = get_calendar_cyclical_features(test_data)
    test_data = pd.concat([test_data, test_cyclical_features], axis=1)
    

    def extract_consec_days(data, day0, n_days, day_size):
        """return n_days consecutive days starting at day0 from feat dataframe"""

        sequence, gt = [], []
        start = day0
        end = start + day_size

        for day in range(n_days):
            sequence.extend(data[[args.trg_feature_name, 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos', 'hour_sin', 'hour_cos']].values[start: end])
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
        datetime_wind = []

        day_idx = 0
        while day_idx < len(data)//day_size - n_days:
            day0 = day_idx*day_size
            sequence, gt = extract_consec_days(data, day0, n_days, day_size)

            time_wind.append(sequence)
            gt_time_wind.append(gt)
            first_date = str(data.index[day0]).replace(':', '')
            last_date = str(data.index[day0 + n_days*day_size-1]).replace(':', '')
            datetime_wind.append(f"{first_date} - {last_date}")

            day_idx += day_stride

        return time_wind, gt_time_wind, datetime_wind


    train_windows, gt_train_windows, date_train_windows = build_dataset(train_data, args.n_days, args.day_size, args.day_stride)
    test_windows, _, date_test_windows = build_dataset(test_data, args.n_days, args.day_size, args.day_stride)

    # save data
    # remove existing files in save target root folder
    existing_files = glob.glob(os.path.join(args.trg_test_save_data, "*", "*.npy"))
    existing_files.extend(glob.glob(os.path.join(args.trg_train_save_data, "*", "*.npy")))
    for f in existing_files:
        os.remove(f)

    # crete save target folders if they don't exist
    os.makedirs(os.path.join(args.trg_test_save_data, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.trg_train_save_data, "data"), exist_ok=True)
    # os.makedirs(os.path.join(args.trg_train_save_data, "gt"), exist_ok=True)

    # save data
    for i, (sample, sample_date) in enumerate(zip(test_windows, date_test_windows)):
        if np.isnan(sample).any(): continue
        np.save(os.path.join(args.trg_test_save_data, "data", sample_date), sample)

    for i, (sample, sample_gt, sample_date) in enumerate(zip(train_windows, gt_train_windows, date_train_windows)):
        if np.isnan(sample).any(): continue
        np.save(os.path.join(args.trg_train_save_data, "data", sample_date), sample)
        # np.save(os.path.join(args.trg_train_save_data, "gt", sample_date), sample_gt)

    # log results
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    print(args, file=open(args.log_file, "a"))
    print(f"Number of train windows: {len(train_windows)}", file=open(args.log_file, "a"))
    print(f"Number of test windows: {len(test_windows)}", file=open(args.log_file, "a"))


if __name__ == "__main__":
    args = parse_args()
    run(args)
    print("Forecasting data ready!")
