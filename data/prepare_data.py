import argparse
import glob
import os
import random

import numpy as np
import pandas as pd

# ---
# Parameters
# ---
parser = argparse.ArgumentParser(description="Prepare data")
parser.add_argument("--csv_data_path", default="data/inpg_dataset/csv_data/", help="Path to CSV data")
parser.add_argument("--feature_name", default="conso_global", help="Name of the feature")
parser.add_argument("--npy_data_path", default="data/inpg_dataset/npy_data/", help="Path to NPY data")

parser.add_argument("--contam_prob", type=float, default=0.05, help="Contamination probability")
parser.add_argument("--min_nbr_anom", type=int, default=5, help="Minimum number of anomalies")
parser.add_argument("--max_nbr_anom", type=int, default=10, help="Maximum number of anomalies")

parser.add_argument("--clean_contam_split", type=float, default=0.1, help="Clean-contamination split ratio")
parser.add_argument("--train_test_split", type=float, default=0.7, help="Train-test split ratio")
parser.add_argument("--window_size", type=int, default=24*3, help="Window size")
parser.add_argument("--step", type=int, default=24, help="Step size")
args = parser.parse_args()

# ---
# Ensure reproductibility
# ---
def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
fix_seeds(0)

# ---
# Loading and train/test splitting
# ---
X = []
csv_paths = glob.iglob(os.path.join(args.csv_data_path, "*.csv"))

for csv_path in csv_paths:
    try: csv_file = pd.read_csv(csv_path)
    except: continue
    
    serie = csv_file[args.feature_name].values
    start, end = 0, args.window_size
    while end<=len(serie):
        sliding_window = serie[start: end]
        start += args.step
        end += args.step
        sliding_window = (sliding_window-sliding_window.mean())/sliding_window.std() # prevents exploding gradients
        X.append(sliding_window)
X = np.stack(X)

N = int(args.clean_contam_split*X.shape[0])
M = int(args.train_test_split*X.shape[0])

clean_data = X[0:N, :]
train_data = X[N:M, :] 
test_data = X[M:, :]   

# ---
# Injecting anomalies into both train and test data
# --- 
# TODO make this a function
train_contam_data = []
train_anom_idx = []
for sample_idx in range(train_data.shape[0]):
    sample = train_data[sample_idx, :].copy()
    peak_value = max(sample.max(), abs(sample.min()))
    is_contam = random.random()<=args.contam_prob
    if is_contam:
        nbr_anom = random.randint(args.min_nbr_anom, args.max_nbr_anom)
        sample_anom_idx = []
        for _ in range(nbr_anom):
            idx = random.randint(0, len(sample)-1)
            sample[idx] = (peak_value+random.random()*peak_value)*-1**random.randint(0, 1)
            sample_anom_idx.append(idx)
        train_contam_data.append(sample)
        train_anom_idx.append(sample_anom_idx)
    else:
        train_contam_data.append(sample)
        train_anom_idx.append([])


test_contam_data = []
test_anom_idx = [] # indices of where the anomalies are for each sequence
for sample_idx in range(test_data.shape[0]):
    sample = test_data[sample_idx, :].copy()
    peak_value = max(sample.max(), abs(sample.min()))
    is_contam = random.random()<=args.contam_prob
    if is_contam:
        sample_anom_idx = []
        nbr_anom = random.randint(args.min_nbr_anom, args.max_nbr_anom)
        for _ in range(nbr_anom):
            idx = random.randint(0, len(sample)-1)
            sample[idx] = (peak_value+random.random()*peak_value)*-1**random.randint(0, 1)
            sample_anom_idx.append(idx)
        test_contam_data.append(sample)
        test_anom_idx.append(sample_anom_idx)
    else:
        test_contam_data.append(sample)
        test_anom_idx.append([])

# ---
# Saving dataset
# ---
for subfolder in ["train/data", "train/gt", "test/data", "test/gt", "clean", "filter", "contam"]:
    folder_tree_path = os.path.join(args.npy_data_path, subfolder)
    try:
        os.makedirs(folder_tree_path, exist_ok=True)
        print(f"Created folder tree: {folder_tree_path}")
    except FileExistsError:
        pass
    except Exception as e:
        print(f"{str(e)}")

def delete_files_in_directory(directory_path):
    try:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
        print(f"All files in '{directory_path}' have been deleted.")
    except Exception as e:
        print(f"{str(e)}")
delete_files_in_directory(args.npy_data_path)


for i, sample in enumerate(clean_data):                                         # clean data to evalute forecasting performance (we don't forecast artificial anomalies)
    if np.isnan(sample).any(): continue
    np.save(os.path.join(args.npy_data_path, "clean", str(i)), sample)

for i, (sample, anom_idx) in enumerate(zip(train_contam_data, train_anom_idx)): # contaminated train data of anomaly detection
    if np.isnan(sample).any(): continue
    np.save(os.path.join(args.npy_data_path, "train", "data", str(i)), sample)
    np.save(os.path.join(args.npy_data_path, "train", "gt", str(i)), anom_idx)

for i, (sample, anom_idx) in enumerate(zip(test_contam_data, test_anom_idx)):   # contaminated test data of anomaly detection
    if np.isnan(sample).any(): continue                                         # also train data of load forecasting (contaminated and filtered)
    np.save(os.path.join(args.npy_data_path, "test", "data", str(i)), sample)
    np.save(os.path.join(args.npy_data_path, "test", "gt", str(i)), anom_idx)

print("new data saved")

