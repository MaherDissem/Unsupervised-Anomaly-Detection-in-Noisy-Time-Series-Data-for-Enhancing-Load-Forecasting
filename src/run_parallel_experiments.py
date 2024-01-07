# TODO increase number of epochs
import os
import inspect
import multiprocessing as mp

from pipeline import run_pipeline


# environment variables
gpu_ids = [0, 1, 2, 3]
nbr_workers = len(gpu_ids)
job_per_gpu = 2

# Grid of experiment parameters, each row is a different experiment, non-specified parameters are set to default
exp_parameters = [
    {"data_folder": "AEMO/NSW",},
    {"data_folder": "AEMO/QLD",},
    {"data_folder": "AEMO/SA",},
    {"data_folder": "AEMO/TAS",},
    {"data_folder": "AEMO/VIC",},
    {"data_folder": "INPG",},
    {"data_folder": "Park/Commercial/30_minutes",},
    {"data_folder": "Park/Office/30_minutes",},
    {"data_folder": "Park/Residential/30_minutes",},
    {"data_folder": "Park/Public/30_minutes",},

    {"data_folder": "AEMO/NSW", "contam_ratio": 0.05},
    {"data_folder": "AEMO/NSW", "contam_ratio": 0.1},
    {"data_folder": "AEMO/NSW", "contam_ratio": 0.15},
    {"data_folder": "AEMO/NSW", "contam_ratio": 0.2},
    {"data_folder": "AEMO/NSW", "contam_ratio": 0.25},

]

def get_default_parameters(params):
    """Set default parameters for non-specified parameters for the experiment"""
    params.setdefault("data_folder", "AEMO/NSW")
    params.setdefault("day_size", 24 if "INPG" in params.get("data_folder", "") else 48)
    params.setdefault("n_days", 1)
    params.setdefault("window_size", params.get("day_size", 24) * params.get("n_days", 1))
    params.setdefault("day_stride", 1)
    params.setdefault("contam_ratio", 0.02 if "INPG" in params.get("data_folder", "") else 0.1)
    params.setdefault("flag_consec", "INPG" not in params.get("data_folder", ""))
    params.setdefault("outlier_threshold", 2.4 if "INPG" in params.get("data_folder", "") else 2.5)
    params.setdefault("forecast_window_size", 6)
    params.setdefault("forecast_day_stride", 1)
    params.setdefault("save_figs", True)
    params.setdefault("imp_trained", False)
    return params

def run_experiment(exp_id, gpu_id, exp_parameters):
    """Run a single experiment with the given parameters on the given GPU"""
    # Set the GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Get the target function's parameter names in the order they are defined
    parameter_order = list(inspect.signature(run_pipeline).parameters.keys())
    
    # Create a dictionary with parameters in the correct order
    exp_parameters.update({"exp_folder": f"exp{exp_id}"})
    ordered_parameters = {param: exp_parameters[param] for param in parameter_order}
    
    # Call run_pipeline with the selected parameters
    run_pipeline(**ordered_parameters)



if __name__ == "__main__":

    # Create a list of arguments for each experiment
    experiment_args = []
    exp = 0
    for exp_id, exp_parameter in enumerate(exp_parameters):
        gpu_id = gpu_ids[exp % len(gpu_ids)]  # Assign GPUs in a round-robin fashion
        exp_parameters = get_default_parameters(exp_parameter)
        experiment_args.append([exp_id, gpu_id, exp_parameters])
        exp += 1

    # Divide the number of experiments by the max number of parallel processes
    chunks = [experiment_args[i:i+nbr_workers*job_per_gpu] for i in range(0, len(experiment_args), nbr_workers*job_per_gpu)]

    # Run the experiments in parallel, one chunk at a time
    for chunk in chunks:
        processes = []
        for args in chunk:
            process = mp.Process(target=run_experiment, args=args)
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

    print("Multi-processes finished")
