# TODO increase number of epochs and early-stop patience
import os
import inspect
import concurrent.futures

from pipeline import run_pipeline


# Environment variables
gpu_ids = [0, 1, 2, 3] # List of GPU IDs to use
job_per_gpu = 3        # Number of jobs to run on each GPU

# Grid of experiment parameters, each row is a different experiment, non-specified parameters are set to default values
exp_parameters = [
    {"data_folder": "AEMO/NSW"},
    {"data_folder": "AEMO/QLD"},
    {"data_folder": "AEMO/SA"},
    {"data_folder": "AEMO/TAS"},
    {"data_folder": "AEMO/VIC"},
    {"data_folder": "INPG"},
    {"data_folder": "Park/Commercial/30_minutes",},
    {"data_folder": "Park/Office/30_minutes",},
    {"data_folder": "Park/Residential/30_minutes",},
    {"data_folder": "Park/Public/30_minutes",},

    {"forecast_window_size": 14, "forecast_sequence_split": 0.5, "data_folder": "AEMO/NSW",},
    {"forecast_window_size": 14, "forecast_sequence_split": 0.5, "data_folder": "AEMO/QLD",},
    {"forecast_window_size": 14, "forecast_sequence_split": 0.5, "data_folder": "AEMO/SA",},
    {"forecast_window_size": 14, "forecast_sequence_split": 0.5, "data_folder": "AEMO/TAS",},
    {"forecast_window_size": 14, "forecast_sequence_split": 0.5, "data_folder": "AEMO/VIC",},
    {"forecast_window_size": 14, "forecast_sequence_split": 0.5, "data_folder": "INPG",},
    {"forecast_window_size": 14, "forecast_sequence_split": 0.5, "data_folder": "Park/Commercial/30_minutes",},
    {"forecast_window_size": 14, "forecast_sequence_split": 0.5, "data_folder": "Park/Office/30_minutes",},
    {"forecast_window_size": 14, "forecast_sequence_split": 0.5, "data_folder": "Park/Residential/30_minutes",},
    {"forecast_window_size": 14, "forecast_sequence_split": 0.5, "data_folder": "Park/Public/30_minutes",},

]

def get_default_parameters(params):
    """ Set default parameters for non-specified experiment parameters.
        Models default parameters are defined in their respective parsing functions.
    """
    params.setdefault("data_folder", "AEMO/NSW")
    params.setdefault("day_size", 24 if "INPG" in params.get("data_folder", "") else 48)
    params.setdefault("n_days", 1)
    params.setdefault("window_size", params.get("day_size", 24) * params.get("n_days", 1))
    params.setdefault("day_stride", 1)
    params.setdefault("day_contam_rate", 0.02 if "INPG" in params.get("data_folder", "") else 0.4)
    params.setdefault("data_contam_rate", 0.05)
    params.setdefault("flag_consec", "INPG" not in params.get("data_folder", ""))
    params.setdefault("outlier_threshold", 2.4 if "INPG" in params.get("data_folder", "") else 2.5)
    params.setdefault("forecast_window_size", 6)
    params.setdefault("forecast_day_stride", 1)
    params.setdefault("forecast_sequence_split", (params.get("forecast_window_size", 6) - 1 )/ params.get("forecast_window_size", 6))
    params.setdefault("forecast_model", "all")
    params.setdefault("save_figs", True)
    return params

def run_experiment(exp_id, gpu_id, exp_parameters):
    """Run a single experiment with the given parameters on the given GPU"""
    # Set the GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Get the target function's parameter names in the order they are defined
    parameter_order = list(inspect.signature(run_pipeline).parameters.keys())
    
    # Create a dictionary with parameters in the correct order
    exp_parameters.update({"exp_folder": f"exp{exp_id}"}) # folder to save data and results in
    ordered_parameters = {param: exp_parameters[param] for param in parameter_order}
    
    # Call run_pipeline with the selected parameters
    run_pipeline(**ordered_parameters)


if __name__ == "__main__":
    n_workers = len(gpu_ids) * job_per_gpu
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for exp_id, exp_parameter in enumerate(exp_parameters):
            # Assign GPUs in a round-robin fashion
            gpu_id = gpu_ids[exp_id % len(gpu_ids)]
            # Get default experiment parameters
            exp_parameters = get_default_parameters(exp_parameter)
            # Submit an experiment for execution and store the future object
            futures[executor.submit(run_experiment, exp_id, gpu_id, exp_parameters)] = (exp_id, gpu_id, exp_parameters)

        # Iterate over completed futures as they become available
        for completed_future in concurrent.futures.as_completed(futures):
            exp_id, gpu_id, exp_parameters = futures[completed_future]
            try:
                completed_future.result() # Blocked until result is ready
            except Exception as e:
                print(f"Experiment {exp_id} on GPU {gpu_id} failed with error: {e}")

    print("Multi-processes finished")
