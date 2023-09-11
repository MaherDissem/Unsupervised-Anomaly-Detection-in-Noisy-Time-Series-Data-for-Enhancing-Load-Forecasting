import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append("load-forecasting/src")
from src.dataset import TS_Dataset
from src.seq2seq import DecoderRNN, EncoderRNN, Net_GRU
from src.train import train_model

import warnings; warnings.simplefilter('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)

# ---
# Parameters
# ---
epochs = 1000
batch_size = 32
lr = 0.001
gamma = 0.01
n_plots = 51

# ---
# Load dataset
# ---
timesteps = 240 
input_target_sample_split_ratio = 0.5834 # split same ts sequence into train and forecasting target
N_input = int(input_target_sample_split_ratio*timesteps) # input length
N_output = timesteps - N_input # target length
train_test_data_split_ratio = 0.8 # split train and test

data = TS_Dataset("data/filtered", "filter", input_target_sample_split_ratio)
# data = TS_Dataset("data/aemo_npy_data", "contam", input_target_sample_split_ratio)

print(f"number of samples: {len(data)}")
train_size = int(train_test_data_split_ratio*len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

trainloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
)
testloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
)

# ---
# train models
# ---
encoder = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, fc_units=16, output_size=1).to(device)
model = Net_GRU(encoder, decoder, target_length=N_output, device=device).to(device)
train_model(trainloader, testloader, model, loss_type='mse', learning_rate=lr, epochs=epochs, device=device)

# ---
# Visualize results
# ---
gen_test = iter(testloader)
test_inputs, test_targets = next(gen_test)
test_inputs  = torch.tensor(test_inputs, dtype=torch.float32).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)

for ind in range(1, n_plots):
    plt.figure()
    plt.rcParams['figure.figsize'] = (10.0, 13.0)  
    k = 1 # loss nbr
    pred = model(test_inputs).to(device)
    input = test_inputs.detach().cpu().numpy()[ind,:,:]
    target = test_targets.detach().cpu().numpy()[ind,:,:]
    preds = pred.detach().cpu().numpy()[ind,:,:]
    plt.subplot(2, 1, k)
    plt.plot(range(0, N_input), input, label='input', linewidth=3)
    plt.plot(range(N_input-1, N_input+N_output), np.concatenate([ input[N_input-1:N_input], target ]), label='target', linewidth=3)   
    plt.plot(range(N_input-1, N_input+N_output),  np.concatenate([ input[N_input-1:N_input], preds ]), label='prediction', linewidth=3)       
    plt.xticks(range(0, 40, 2))
    plt.legend()
    k += 1
    # plt.show()
    plt.savefig(f"out_figs/{ind}.jpg")