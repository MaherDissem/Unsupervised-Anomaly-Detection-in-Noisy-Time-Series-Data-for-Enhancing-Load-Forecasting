import multiprocessing as mp
import os
import subprocess

# environment variables
gpu_ids = [0, 1, 2, 3]
nbr_workers = len(gpu_ids)

# experiment variables
experiments_root = "experiments2"
datasets = ["aemo_dataset", "inpg_dataset", "IRISE_dataset"]
feature_names = ["TOTALDEMAND", "conso_global", "Site consumption ()"]
window_sizes = [45, 24, 50]
nbr_days = [2, 3, 5] # multiplier of window size and sliding step
contam_rates = [0.05, 0.1, 0.15, 0.2]
sequence_splits = [0.5, 0.75, 0.9]

def run_experiment(exp, dataset, feature_name, contam_rate, min_nbr_anom, max_nbr_anom, timesteps, step, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    exp_folder = os.path.join(experiments_root, f"exp{exp}_{dataset}")
    results_path = os.path.join(exp_folder, "results.txt")
    os.makedirs(exp_folder, exist_ok=True)
    print(f"exp: {exp}, dataset: {dataset}, contam_rate: {contam_rate}, timesteps: {timesteps}",\
          file=open(results_path, "a"))

    # prepare data
    command = [
        "python", "data/prepare_data.py",
        "--csv_data_path", f"data/{dataset}/csv_data/",
        "--feature_name", f"{feature_name}",
        "--npy_data_path", f"{exp_folder}/data",
        "--contam_prob", str(contam_rate),
        "--min_nbr_anom", str(min_nbr_anom),
        "--max_nbr_anom", str(max_nbr_anom),
        "--window_size", str(timesteps),
        "--step", str(step)
    ]
    subprocess.run(command)

    # train feature extractor
    command = [
        "python", "anomaly-detection/src/train_feature_extractor.py",
        "--dataset_root", f"{exp_folder}/data",
        "--path_to_save_plot",  f"{exp_folder}/out_figs",
        "--checkpoint_path", f"{exp_folder}/checkpoint.pt",
        "--seq_len", str(timesteps),
        "--nbr_variables", "1",
        "--embedding_dim", str(timesteps),
        "--epochs", "200",
        "--every_epoch_print", "200",
        "--patience", "25",
        "--results_file", str(results_path),
    ]
    subprocess.run(command)

    # train anomaly detector and save filtered data
    command = [
        "python", "anomaly-detection/main.py",
        "--filtered_data_path", f"{exp_folder}/data/filter",
        "--data_path", f"{exp_folder}/data",
        "--extractor_weights", f"{exp_folder}/checkpoint.pt",
        "--nbr_timesteps", str(timesteps),
        "--extractor_embedding_dim", str(timesteps),
        "--results_file", str(results_path),
    ]
    subprocess.run(command)

    for sequence_split in sequence_splits:
        forecast_horizon = int(sequence_split * timesteps)
        print(f"\n-----\nForecast horizon of {forecast_horizon} ({sequence_split} split)",\
            file=open(results_path, "a"))
        
        # train load forecasting model on contam data
        command = [
            "python", "load-forecasting/main.py",
            "--train_dataset_path", f"{exp_folder}/data/test/data",
            "--test_dataset_path", f"{exp_folder}/data/clean",
            "--loss_type", "mse",
            "--timesteps", str(timesteps),
            "--epochs", "200",
            "--sequence_split", str(sequence_split),
            "--save_plots_path", f"{exp_folder}/out_figs/{forecast_horizon}/contam",
            "--results_file", str(results_path),
        ]
        print(f"\nContam data forecasting",\
            file=open(results_path, "a"))
        subprocess.run(command)

        # train load forecasting model on filtered data
        command = [
            "python", "load-forecasting/main.py",
            "--train_dataset_path", f"{exp_folder}/data/filter",
            "--test_dataset_path", f"{exp_folder}/data/clean",
            "--loss_type", "mse",
            "--timesteps", str(timesteps),
            "--epochs", "200",
            "--sequence_split", str(sequence_split),
            "--save_plots_path", f"{exp_folder}/out_figs/{forecast_horizon}/filter",
            "--results_file", str(results_path),
        ]
        print(f"\nFilter data forecasting",\
        file=open(results_path, "a"))
        subprocess.run(command)


def create_clean_folder(path):
    os.makedirs(path, exist_ok=True)
    try:
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    create_clean_folder(experiments_root)
    experiment_args = []
    exp = 0
    for dataset, feature_name, window_size in zip(datasets, feature_names, window_sizes):
        for nbr_day in nbr_days:
            timesteps = nbr_day * window_size
            step = window_size
            for contam_rate in contam_rates:
                min_nbr_anom = int(contam_rate * timesteps)
                max_nbr_anom = int(contam_rate * timesteps+1)
                gpu_id = gpu_ids[exp % len(gpu_ids)]  # Assign GPUs in a round-robin fashion
                experiment_args.append((exp, dataset, feature_name, contam_rate, min_nbr_anom, max_nbr_anom, timesteps, step, gpu_id))
                exp += 1
    
    # divide the number of experiments by the max number of parallel processes
    chunks = [experiment_args[i:i+nbr_workers] for i in range(0, len(experiment_args), nbr_workers)]

    for chunk in chunks:
        processes = []
        for args in chunk:
            process = mp.Process(target=run_experiment, args=args)
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

    print("Done!")

