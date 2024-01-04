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


# load parameters
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for anomaly detection model training and evaluation.")
    parser.add_argument("--raw_data_csv",        type=str,   default="dataset/raw/INPG/predis-mhi.csv", help="Path to raw data root")
    parser.add_argument("--trg_save_data",        type=str,   default="dataset/processed/INPG", help="Path to save processed data")
    parser.add_argument("--load_feature_name",    type=str,   default="conso_global", help="Name of the load feature")
    parser.add_argument("--date_feature_name",    type=str,   default="date_time", help="Name of the date_time feature")
    parser.add_argument("--day_size",             type=int,   default=24, help="Size of a day")
    parser.add_argument("--n_days",               type=int,   default=1, help="Number of days")
    parser.add_argument("--day_stride",           type=int,   default=1, help="Day stride for sliding window")
    parser.add_argument("--contam_clean_ratio",   type=float, default=0.8, help="Clean data save ratio (forcasting model is later evaluated on this clean data)")
    parser.add_argument("--ad_split_ratio",       type=float, default=0.7, help="Anomaly detection train-test split ratio")
    parser.add_argument("--seed",                 type=int,   default=0, help="Random seed")
    parser.add_argument("--log_file",             type=str,   default="results/results.txt", help="Path of file to log to") # quantiles must be saved to later scale back metrics
    args = parser.parse_args()
    return args


def run(args):
    set_seed(args.seed)
    load = pd.read_csv(args.raw_data_csv, sep=";")
    load['date_time'] = pd.to_datetime(load['date_time'], format="%m/%d/%Y %H:%M:%S")
    load.set_index(args.date_feature_name, inplace=True)

    # remove duplicate indices
    load = load[~load.index.duplicated()]

    # replace missing values (if any) by the value of the previous week
    idx = pd.date_range(load.index[0], load.index[-1], freq="H")
    load = load.reindex(idx, fill_value=np.nan)
    load = load.fillna(load.shift(args.day_size*7))

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
    lab_holidays = []
    for year in range(2016, 2023):
        lab_holidays.extend(pd.date_range(f'{year}-07-31', f'{year}-08-16', freq='H')[:-1]) # summer holidays
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

    to_remove = corrupt_data_dates + covid_dates + lab_holidays + full_Holiday_date
    # load = load[~load.index.isin(to_remove)]

    # remove uncomplete last day
    uncomplete_last_day = load[load.index.day == load.index[-1].day]
    load = load.drop(uncomplete_last_day.index)
        
    # split contam data into train and test sets for anomaly detection model
    N = int(args.contam_clean_ratio*len(load))//args.day_size*args.day_size
    M = int(args.ad_split_ratio*(len(load[:N])))//args.day_size*args.day_size
    contaminated_load = load[:N].copy()
    contaminated_load = contaminated_load[~contaminated_load.index.isin(to_remove)] # remove corrupt data to train AD/AI model
    clean_load = load[N:]
    ad_train_load = contaminated_load[:M]
    ad_test_load = contaminated_load[M:]

    def extract_consec_days(load, day0, n_days, day_size):
        """return n_days consecutive days starting at day0 from load dataframe"""

        sequence = []
        start = day0
        end = start + day_size

        for day in range(n_days):
            sequence.extend(load[args.load_feature_name].values[start: end])
            start += day_size
            end += day_size
        return np.array(sequence)

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
            sequence = extract_consec_days(load, day0, n_days, day_size)

            time_wind.append(sequence)
            first_date = str(load.index[day0]).replace(':', '')
            last_date = str(load.index[day0 + n_days*day_size-1]).replace(':', '')
            datetime_wind.append(f"{first_date} - {last_date}")
            day_idx += day_stride

        return time_wind, datetime_wind

    ad_train_windows, date_ad_train_windows = build_dataset(ad_train_load, args.n_days, args.day_size, args.day_stride)
    ad_test_windows, date_ad_test_windows = build_dataset(ad_test_load, args.n_days, args.day_size, args.day_stride)

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

    # crete save target folders if they don't exist
    os.makedirs(os.path.join(args.trg_save_data, "lf_test_clean", "data"), exist_ok=True)
    os.makedirs(os.path.join(args.trg_save_data, "ad_train_contam", "data"), exist_ok=True)
    os.makedirs(os.path.join(args.trg_save_data, "ad_test_contam", "data"), exist_ok=True)

    # save contam ad train data
    for i, (sample, sample_date) in enumerate(zip(ad_train_windows, date_ad_train_windows)):
        if np.isnan(sample).any(): continue
        np.save(os.path.join(args.trg_save_data, "ad_train_contam", "data", sample_date), sample)

    # save contam ad test data
    for i, (sample, sample_date) in enumerate(zip(ad_test_windows, date_ad_test_windows)):
        if np.isnan(sample).any(): continue
        np.save(os.path.join(args.trg_save_data, "ad_test_contam", "data", sample_date), sample)

    # log results
    print(args, file=open(args.log_file, "a"))
    print(f"Number of ad_train_contam windows: {len(ad_train_windows)}", file=open(args.log_file, "a"))
    print(f"Number of ad_test_contam windows: {len(ad_test_windows)}", file=open(args.log_file, "a"))

    print(f"min_quantile={min_quantile:0.3f} -> value={min_q_val}", file=open(args.log_file, "a"))
    print(f"max_quantile={max_quantile:0.3f} -> value={max_q_val}", file=open(args.log_file, "a"))

    # # replace outliers by the value of the previous week in clean load, for a fair forecasting model evaluation
    # outliers = clean_load[clean_load[args.load_feature_name] > 2*clean_load[args.load_feature_name].quantile(0.99)]
    # for i in range(len(outliers)):
    #     clean_load.loc[outliers.index[i], args.load_feature_name] = clean_load.loc[outliers.index[i] - pd.Timedelta(weeks=1), args.load_feature_name]

    # save clean load for forecasting model evaluation
    clean_load = (clean_load - min_q_val) / (max_q_val - min_q_val)
    clean_load.rename_axis("date", inplace=True)
    clean_load.to_csv(os.path.join(args.trg_save_data, "load_clean_lf_test.csv"))

    # save contaminated load serie to infer AD/AI models after training
    scaled_load = (load - min_q_val) / (max_q_val - min_q_val)
    scaled_load.rename_axis("date", inplace=True)
    scaled_load.to_csv(os.path.join(args.trg_save_data, "load_contam.csv"))
    print('Dataset ready!')

    return min_q_val, max_q_val


if __name__ == "__main__":
    args = parse_args()
    run(args)
