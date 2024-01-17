import argparse
import os
import glob
import numpy as np
import pandas as pd
import holidays
from datetime import datetime, timedelta

import sys
sys.path.append("./src")

from utils.utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for anomaly detection model training and evaluation.")
    parser.add_argument("--raw_train_data_csv",   type=str,   default="dataset/processed/INPG/load_cleaned.csv", help="Path to raw data root")
    parser.add_argument("--trg_train_save_data",  type=str,   default="dataset/processed/INPG/lf_cleaned", help="Path to save processed data")
    
    parser.add_argument("--raw_test_data_csv",    type=str,   default="dataset/processed/INPG/load_clean_lf_test.csv", help="Path to raw data root")
    parser.add_argument("--trg_test_save_data",   type=str,   default="dataset/processed/INPG/lf_test_clean", help="Path to save processed data")
    
    parser.add_argument("--trg_feature_name",     type=str,   default="conso_global", help="Name of the feat feature")
    parser.add_argument("--date_feature_name",    type=str,   default="date", help="Name of the date_time feature")

    parser.add_argument("--day_size",             type=int,   default=24, help="Size of a day")
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

    # removing national holidays
    fr_holidays = holidays.France(years=list(range(2016, 2023)))
    holiday_dates = list(fr_holidays.keys())
    sampled_dates = []
    for date in holiday_dates:
        start_time = datetime(date.year, date.month, date.day, 0, 0, 0) # Start at 00:00:00
        sampled_dates.append(start_time)
        for _ in range(23):
            start_time += timedelta(hours=1)
            sampled_dates.append(start_time)
    full_Holiday_date = [str(date) for date in sampled_dates]

    # remove lab holidays
    # TODO switch to 1D frequency instead of hourly
    lab_holidays = []
    for year in range(2016, 2023):
        lab_holidays.extend(pd.date_range(f'{year}-07-31', f'{year}-08-16', freq='H')[:-1])   # summer holidays
        lab_holidays.extend(pd.date_range(f'{year}-12-21', f'{year+1}-01-04', freq='H')[:-1]) # christmas holidays

    # remove covid period
    covid_dates = list(pd.date_range('2020-03-17', '2020-06-12', freq='H')[:-1])

    # remove corrupt data
    corrupt_data_dates = []
    # date range
    corrupt_data_dates.extend(pd.date_range('2016-06-21', '2016-11-11', freq="H")[:-1])
    corrupt_data_dates.extend(pd.date_range('2017-08-16', '2017-08-21', freq="H")[:-1])
    corrupt_data_dates.extend(pd.date_range('2019-07-29', '2019-08-23', freq="H")[:-1])
    corrupt_data_dates.extend(pd.date_range('2020-09-21', '2020-10-08', freq="H")[:-1])
    corrupt_data_dates.extend(pd.date_range('2021-05-14', '2021-05-17', freq="H")[:-1])
    # 1 day
    corrupt_data_dates.extend(pd.date_range('2019-05-31', '2019-06-01', freq="H")[:-1])
    corrupt_data_dates.extend(pd.date_range('2019-08-23', '2019-08-24', freq="H")[:-1])
    corrupt_data_dates.extend(pd.date_range('2022-05-27', '2022-05-28', freq="H")[:-1])

    # empty days
    empty_days = []
    days_in_train_index = pd.date_range(train_data.index[0], train_data.index[-1], freq="D")
    for day in days_in_train_index:
        date = str(day).split(' ')[0]
        if len(train_data.loc[date]) == 0:
            next_date = str(day + timedelta(days=1)).split(' ')[0]
            hours = pd.date_range(date, next_date, freq="H")
            empty_days.extend(hours)
    days_in_test_index = pd.date_range(test_data.index[0], test_data.index[-1], freq="D")
    for day in days_in_test_index:
        date = str(day).split(' ')[0]
        if len(test_data.loc[date]) == 0:
            next_date = str(day + timedelta(days=1)).split(' ')[0]
            hours = pd.date_range(date, next_date, freq="H")
            empty_days.extend(hours)

    to_remove = corrupt_data_dates + covid_dates + lab_holidays + full_Holiday_date + empty_days
    days_to_remove = []
    for date in to_remove:
        days_to_remove.append(str(date).split(' ')[0])
    days_to_remove = sorted(list(set(days_to_remove)))

    def extract_consec_days(load, day0, n_days, day_size):
        """return n_days consecutive days starting at day0 from load dataframe"""

        # discard feeding time windows that contain corrupt data to the model
        start_date = str(load.index[day0]).split(' ')[0]
        end_date = str(load.index[day0 + day_size*n_days-1]).split(' ')[0]
        days_in_seq = pd.date_range(start_date, end_date, freq="D")
        days_in_seq = [str(day).split(' ')[0] for day in days_in_seq]
        for day in days_in_seq: 
            if day in days_to_remove:
                return None, start_date, end_date

        sequence = []
        start = day0
        end = start + day_size

        for day in range(n_days):
            sequence.extend(load[args.trg_feature_name].values[start: end])
            start += day_size
            end += day_size

        return np.array(sequence), start_date, end_date

    def build_dataset(load, n_days, day_size, day_stride):
        """
            build a dataset from load dataframe using a sliding window of size n_days and stride of 1 day 
            while contamining the data with synthetic anomalies
        """

        time_wind = []
        datetime_wind = []

        day_idx = 0
        while day_idx < len(load)//day_size - n_days:
            day0 = day_idx*day_size
            sequence, first_date, last_date = extract_consec_days(load, day0, n_days, day_size)

            if sequence is None:
                day_idx += day_stride
                continue

            time_wind.append(sequence)
            datetime_wind.append(f"{first_date} - {last_date}")
            day_idx += day_stride

        return time_wind, datetime_wind

    train_windows, date_train_windows = build_dataset(train_data, args.n_days, args.day_size, args.day_stride)
    test_windows, date_test_windows = build_dataset(test_data, args.n_days, args.day_size, args.day_stride)
    
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

    for i, (sample, sample_date) in enumerate(zip(train_windows, date_train_windows)):
        if np.isnan(sample).any(): continue
        np.save(os.path.join(args.trg_train_save_data, "data", sample_date), sample)

    # log results
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    print(args, file=open(args.log_file, "a"))
    print(f"Number of train windows: {len(train_windows)}", file=open(args.log_file, "a"))
    print(f"Number of test windows: {len(test_windows)}", file=open(args.log_file, "a"))


if __name__ == "__main__":
    args = parse_args()
    run(args)
    print("Forecasting data ready!")

