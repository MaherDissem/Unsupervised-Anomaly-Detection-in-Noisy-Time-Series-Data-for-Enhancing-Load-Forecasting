import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# plots_path = "dataset/processed/AEMO/SA/lf_test_clean/data"
# plots_path = "dataset/processed/AEMO/SA/lf_train_contam/data"
plots_path = "dataset/processed/AEMO/SA/lf_train_filter/data"

os.makedirs(os.path.join(plots_path, "..", "images"), exist_ok=True)
for jpg_path in  glob.iglob(os.path.join(plots_path, "..", "images", "*.jpg")):
    os.remove(jpg_path)

for npy_path in  tqdm(glob.glob(os.path.join(plots_path, "*.npy"))):
    data = np.load(npy_path)
    plt.plot(data)
    plt.title(npy_path)
    plt.savefig(os.path.join(plots_path, "..", "images", os.path.basename(npy_path).replace(".npy", ".jpg")))
    plt.clf()

