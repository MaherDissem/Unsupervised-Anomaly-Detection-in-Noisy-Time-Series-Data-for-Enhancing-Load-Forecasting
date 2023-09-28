import os
from glob import iglob
import numpy as np


def collect_results(root_path):
    exp_params = {}
    results = []
    for path in iglob(os.path.join(root_path, '*', 'results.txt'), recursive=True):
        try:
            with open(path, 'r') as f:
                file_content = f.readlines()

                exp_i = file_content[0].split(' ')[1][:-1]
                dataset = file_content[0].split(' ')[3][:-1]
                contam_rate = file_content[0].split(' ')[5][:-1]
                timesteps = file_content[0].split(' ')[7][:-1]
                recons_loss = file_content[2].split(' ')[3][:4]
                anom_det_f1 = file_content[5].split(' ')[4]
                anom_det_auroc = file_content[5].split(' ')[7][:-1]
                
                i = 8; j=0
                while i<len(file_content):
                    if exp_i == "1" and j == 2: 
                         print("f")
                    horizon = file_content[i].split(' ')[3]
                    split_ratio = file_content[i].split(' ')[4][1:5]
                    
                    contam_mse_loss = file_content[i+3].split(' ')[2][:6]
                    #mae, mape added
                    contam_dtw_loss = file_content[i+3].split(' ')[4][:6]
                    contam_dti_loss = file_content[i+3].split(' ')[6][:6]

                    filter_mse_loss = file_content[i+6].split(' ')[2][:6]
                    filter_dtw_loss = file_content[i+6].split(' ')[4][:6]
                    filter_dti_loss = file_content[i+6].split(' ')[6][:6]

                    exp_params[f"exp_{exp_i}_{j}"] = {
                        "dataset": dataset,
                        "contam_rate": contam_rate,
                        "timesteps": timesteps,
                        "recons_loss": recons_loss,
                        "anom_det_f1": anom_det_f1,
                        "anom_det_auroc": anom_det_auroc,
                        "horizon": horizon,
                        "split_ratio": split_ratio,
                        "contam_mse_loss": contam_mse_loss,
                        "contam_dtw_loss": contam_dtw_loss,
                        "contam_dti_loss": contam_dti_loss,
                        "filter_mse_loss": filter_mse_loss,
                        "filter_dtw_loss": filter_dtw_loss,
                        "filter_dti_loss": filter_dti_loss
                    }
                    results.append((np.round(float(contam_mse_loss) - float(filter_mse_loss), 4), f"exp_{exp_i}_{j}")) # contam - filter: loss decrese after filtering
                    i += 9; j += 1
        except:
            print(f"Error in {path}")
    return results, exp_params

if __name__ == "__main__":
    results, exp_params = collect_results('experiments')
    results.sort(key=lambda x: x[0], reverse=True)
    for result in results[:6]: 
        print("loss decrease after filter", result[0])
        print("exp id:", result[1])
        print("experiment parameters", exp_params[result[1]], end="\n\n")
