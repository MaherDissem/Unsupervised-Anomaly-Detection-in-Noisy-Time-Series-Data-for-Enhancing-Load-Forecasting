import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# dataset_root = "dataset/processed/AEMO/test"
# paths = [
#     f"{dataset_root}/lf_test_clean/data",
#     f"{dataset_root}/ad_train_contam/data",
#     f"{dataset_root}/ad_test_contam/data",
#     # f"{dataset_root}/lf_train_contam/data",
#     # f"{dataset_root}/lf_test_clean/data",
#     # f"{dataset_root}/ai_train/data",
# ]

# paths = ["dataset/processed/AEMO/test/ai_train/data"]

# paths = ["dataset/processed/Yahoo/A1Benchmark/ad_test_contam/data", "dataset/processed/Yahoo/A1Benchmark/ad_train_contam/data"]
# gt_paths = ["dataset/processed/Yahoo/A1Benchmark/ad_test_contam/gt", "dataset/processed/Yahoo/A1Benchmark/ad_train_contam/gt"]
paths = ["dataset/processed/Park/Office/30_minutes/ad_test_contam",]# "dataset/processed/AEMO/NSW/lf_contam/data"]:

for plots_path in paths:
    os.makedirs(os.path.join(plots_path, "..", "images"), exist_ok=True)
    for jpg_path in  glob.iglob(os.path.join(plots_path, "..", "images", "*.jpg")):
        os.remove(jpg_path)

    # for npy_path, gt_path in tqdm(zip(glob.glob(os.path.join(plots_path, "*.npy")))), glob.glob(os.path.join(gt_paths[paths.index(plots_path)], "*.npy")))):
    #     gt = np.load(gt_path)
    #     if gt.sum() == 0: continue
    #     plt.plot(gt, label="gt")
    for npy_path in tqdm(glob.glob(os.path.join(plots_path, "*.npy"))):
        data = np.load(npy_path)
        plt.plot(data, label="data")
        plt.legend()
        plt.title(npy_path)
        plt.savefig(os.path.join(plots_path, "..", "images", os.path.basename(npy_path).replace(".npy", ".jpg")))
        plt.clf()

