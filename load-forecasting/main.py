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
epochs = 25
batch_size = 32
lr = 0.001
gamma = 0.01
n_plots = 32 # must be <= batch_size, TODO fix this

# ---
# Load data
# ---
timesteps = 240 
sequence_split = 0.5834 # split same ts sequence into train and forecasting target
N_input = int(sequence_split*timesteps) # input length
N_output = timesteps - N_input # target length

# train_data = TS_Dataset("data/aemo_npy_data", "test", sequence_split) # contam test data
train_data = TS_Dataset("data/filtered", "filter", sequence_split) # contam test data filltered through anomaly detection
test_data = TS_Dataset("data/aemo_npy_data", "train", sequence_split) # forecast target should be anomaly free, otherwise metric is not fair

trainloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
)
testloader = DataLoader(
    test_data,
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
inputs, targets = next(gen_test)
inputs  = torch.tensor(inputs, dtype=torch.float32).to(device)
targets = torch.tensor(targets, dtype=torch.float32).to(device)
preds = model(inputs).to(device)

for ind in range(1, n_plots):
    plt.figure()
    plt.rcParams['figure.figsize'] = (10.0, 5.0)  
    input = inputs.detach().cpu().numpy()[ind,:,:]
    target = targets.detach().cpu().numpy()[ind,:,:]
    pred = preds.detach().cpu().numpy()[ind,:,:]
    plt.plot(range(0, N_input), input, label='input', linewidth=3)
    plt.plot(range(N_input-1, N_input+N_output), np.concatenate([input[N_input-1:N_input], target]), label='target', linewidth=3)   
    plt.plot(range(N_input-1, N_input+N_output),  np.concatenate([input[N_input-1:N_input], pred]), label='prediction', linewidth=3)       
    plt.legend()
    plt.savefig(f"results/out_figs/filtered/{ind}.jpg")
