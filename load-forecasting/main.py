import argparse
import os
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

# ---
# Parameters
# ---
parser = argparse.ArgumentParser(description="Runs Load Forecasting experiments")
# dataset
parser.add_argument("--train_dataset_path", default="data/inpg_dataset/npy_data/test/data", help="Path to train dataset")
parser.add_argument("--test_dataset_path", default="data/inpg_dataset/npy_data/clean", help="Path to clean dataset")
# sequence
parser.add_argument("--timesteps", type=int, default=24*3, help="Number of timesteps")
parser.add_argument("--nbr_var", type=int, default=1, help="Number of variables")
parser.add_argument("--sequence_split", type=float, default=0.5834, help="Sequence split ratio")
# model
parser.add_argument("--loss_type", type=str, default="mse", help="Loss function to optimize (mse/dilate)")
parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the model")
parser.add_argument("--num_grulstm_layers", type=int, default=1, help="Number of GRU/LSTM layers")
parser.add_argument("--fc_units", type=int, default=16, help="Number of fully connected units")
# training
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.01, help="Gamma parameter")
# visualization
parser.add_argument("--n_plots", type=int, default=32, help="Number of plots")
parser.add_argument("--save_plots_path", default="results/out_figs/inpg_dataset/contam", help="Path to save plots")
parser.add_argument("--results_file", default="results/results.txt", help="Path to file to save results in")
args = parser.parse_args()

# ---
# Ensure reproductibility
# ---
def fix_seeds(seed, with_torch=True, with_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
fix_seeds(0)

# ---
# Load data
# ---
N_input = int(args.sequence_split*args.timesteps)  # input length
N_output = args.timesteps - N_input                # target length

train_data = TS_Dataset(args.train_dataset_path, ts_split=args.sequence_split)  
test_data = TS_Dataset(args.test_dataset_path, ts_split=args.sequence_split)   # forecast target should be anomaly free, otherwise metric is not fair

trainloader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
)
testloader = DataLoader(
    test_data,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
) 

# ---
# train models
# ---
encoder = EncoderRNN(
    input_size=args.nbr_var, 
    hidden_size=args.hidden_size, 
    num_grulstm_layers=args.num_grulstm_layers, 
    batch_size=args.batch_size
).to(device)

decoder = DecoderRNN(
    input_size=args.nbr_var, 
    hidden_size=args.hidden_size, 
    num_grulstm_layers=args.num_grulstm_layers, 
    fc_units=args.fc_units, 
    output_size=args.nbr_var
).to(device)

model = Net_GRU(encoder, decoder, target_length=N_output, device=device).to(device)
train_loss_evol = train_model(trainloader, testloader, model, loss_type=args.loss_type, learning_rate=args.lr, epochs=args.epochs, device=device, log_file=args.results_file, gamma=args.gamma, verbose=1)

# ---
# Visualize results
# ---
plt.plot(train_loss_evol)
os.makedirs(args.save_plots_path, exist_ok=True)
plt.savefig(args.save_plots_path + "/forecast_train_loss_evol.jpg")

gen_test = iter(testloader) 
inputs, targets = next(gen_test)
inputs  = torch.tensor(inputs, dtype=torch.float32).to(device)
targets = torch.tensor(targets, dtype=torch.float32).to(device)
preds = model(inputs).to(device)

for ind in range(1, min(args.batch_size, args.n_plots)):
    plt.figure()
    plt.rcParams['figure.figsize'] = (10.0, 5.0)  
    input = inputs.detach().cpu().numpy()[ind,:,:]
    target = targets.detach().cpu().numpy()[ind,:,:]
    pred = preds.detach().cpu().numpy()[ind,:,:]
    plt.plot(range(0, N_input), input, label='input', linewidth=3)
    plt.plot(range(N_input-1, N_input+N_output), np.concatenate([input[N_input-1:N_input], target]), label='target', linewidth=3)   
    plt.plot(range(N_input-1, N_input+N_output),  np.concatenate([input[N_input-1:N_input], pred]), label='prediction', linewidth=3)       
    plt.legend()
    plt.savefig(f"{args.save_plots_path}/{ind}.jpg")