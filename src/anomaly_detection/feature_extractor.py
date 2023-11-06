import os
import sys

import torch
import torch.nn as nn
import numpy as np
import statsmodels.api as sm

sys.path.insert(0, os.getcwd()) 
from src.utils.early_stop import EarlyStopping


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seasonal_decomposition(input_data, seasonal_period=48, dim=-1):
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
            time_series = input_data[i, :, j].cpu().numpy()
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


def gen_ts_features(input_data):
    """Generate time series features from a batch of time series data."""
    # torch.Size([32, 144, 1]) -> torch.Size([32, 3, 144, 1])
    
    batch_size, sequence_length, input_dim = input_data.size()
    input_data = input_data.to(device)
    
    seasonal_components = seasonal_decomposition(input_data, seasonal_period=48)
    moving_averages = moving_average(input_data, alpha=0.2)

    return torch.stack([input_data, seasonal_components, moving_averages], dim=1)
