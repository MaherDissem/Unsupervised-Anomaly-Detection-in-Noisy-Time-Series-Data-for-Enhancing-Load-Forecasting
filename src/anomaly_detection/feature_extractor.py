import torch
import numpy as np
import statsmodels.api as sm


def seasonal_decomposition(input_data, seasonal_period, dim=-1, device=torch.device("cpu")):
    """
    Perform seasonal decomposition of a batch of time series and extract the seasonal component using PyTorch.

    Args:
    - input_data (torch.Tensor): The batch of time series data (3D tensor).
    - seasonal_period (int): The seasonal period of the time series.
    - dim (int): The dimension along which to insert the components (default: -1).

    Returns:
    - seasonal_components (torch.Tensor): The seasonal component of the batch of time series.
    """
    batch_size, sequence_length, input_dim = input_data.size()
    data_components = []

    for i in range(batch_size):
        batch_components = []

        for j in range(input_dim):
            time_series = input_data[i, :, j].cpu().numpy() # this slows down computation
            decomposition = sm.tsa.seasonal_decompose(time_series, model='additive', period=seasonal_period)
            seasonal_component = torch.tensor(decomposition.seasonal, dtype=torch.float32, device=device)

            batch_components.append(seasonal_component)
        batch_components = torch.stack(batch_components, dim=0)

        data_components.append(batch_components)
    data_components = torch.stack(data_components, dim=0)

    return data_components.permute(0, 2, 1)


def moving_average(input_data, alpha=0.2):
    """
    Calculate a custom moving average with a specified alpha for a batch of time series.

    Args:
    - input_data (numpy.ndarray): The batch of time series data (3D array).
    - alpha (float): Smoothing factor, typically between 0 and 1.

    Returns:
    - moving_average (numpy.ndarray): The custom moving average of the batch of time series.
    """
    batch_size, sequence_length, input_dim = input_data.shape
    moving_average = torch.zeros_like(input_data, dtype=torch.float32)

    for i in range(batch_size):
        for j in range(input_dim):
            moving_average[i, :, j] = input_data[i, :, j]
            for t in range(1, sequence_length):
                moving_average[i, t, j] = alpha * input_data[i, t, j] + (1 - alpha) * moving_average[i, t - 1, j]

    return moving_average


def patch_std_1d(time_series, patch_size):
    """Calculate the standard deviation of each patch of size patch_size in the time series with no overlap.
    Args:
        time_series (np.array): Time series to calculate the patch standard deviation.
        patch_size (int): Size of the patch.
    Returns:
        np.array: Patch standard deviation of the time series.
    """
    num_patches = len(time_series) // patch_size
    patches_std = torch.zeros_like(time_series)
    for i in range(num_patches):
        start_idx = i * patch_size
        end_idx = (i + 1) * patch_size
        patch = time_series[start_idx:end_idx]
        patches_std[start_idx: end_idx] = torch.std(patch)

    return patches_std


def patch_std(input_data, window_size=8):
    """Calculate the patch standard deviation of a batch of time series data.
    Args:
        input_data (np.array): Batch of time series data.
        window_size (int): Size of the patch.
    Returns:
        np.array: Patch standard deviation of the batch of time series.
    """
    batch_size, sequence_length, input_dim = input_data.shape
    rolling_std = torch.zeros_like(input_data, dtype=torch.float32)

    for batch in range(batch_size):
        for dim in range(input_dim):
            timeseries = input_data[batch, :, dim]
            rolling_std[batch, :, dim] = patch_std_1d(timeseries, window_size)
    
    return rolling_std


def gen_ts_features(input_data, feat_patch_size, alpha):
    """Generate time series features from a batch of time series data."""
    # batch_size, sequence_length, input_dim = input_data.size()
    # torch.Size([32, 144, 1]) -> torch.Size([32, 3, 144, 1])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    
    # seasonal_components = seasonal_decomposition(input_data, seasonal_period, device=device)
    patched_std = patch_std(input_data, feat_patch_size)
    moving_averages = moving_average(input_data, alpha=alpha)

    # return torch.stack([input_data, seasonal_components, moving_averages], dim=1)
    return torch.stack([input_data, patched_std, moving_averages], dim=1)
