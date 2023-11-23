import os
import subprocess

save_folder = "dataset/raw/AEMO"

for location in ["NSW", "QLD", "VIC", "SA", "TAS"]:
    for year in range(2015, 2020+1): # sampling rate for 2021+ is different
        for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
            file_save_path = f"{save_folder}/{location}/{year}{month}.csv"
            os.makedirs(os.path.dirname(file_save_path), exist_ok=True)
            bash_command = f"curl https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{year}{month}_{location}1.csv > {file_save_path}"
            result = subprocess.run(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print("Command failed:")
                print(result.stderr)
            else:
                print(f"Downloaded {year}-{month}-{location}.csv")
