import sys
import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset_f import F_Dataset
from model import DecoderRNN, EncoderRNN, Net_GRU
from train import train_model

sys.path.insert(0, os.getcwd())
from src.utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Runs Load Forecasting experiments")
    # dataset
    parser.add_argument("--train_dataset_path",   type=str,       default="dataset/processed/INPG/lf_cleaned",       help="Path to train dataset") # dataset parameter
    parser.add_argument("--test_dataset_path",    type=str,       default="dataset/processed/INPG/lf_test_clean",    help="Path to clean dataset for testing") # dataset parameter
    # sequence
    parser.add_argument("--timesteps",            type=int,       default=24*6,     help="Number of timesteps")    # dataset parameter
    parser.add_argument("--sequence_split",       type=float,     default=5/6,      help="Sequence split ratio")   # dataset parameter
    parser.add_argument("--nbr_var",              type=int,       default=1,        help="Number of variables")
    # model parameters
    parser.add_argument("--hidden_size",          type=int,       default=128,      help="Hidden size of the model")
    parser.add_argument("--num_grulstm_layers",   type=int,       default=1,        help="Number of GRU/LSTM layers")
    parser.add_argument("--fc_units",             type=int,       default=16,       help="Number of fully connected units")
    # training
    parser.add_argument("--epochs",               type=int,       default=1,      help="Number of epochs") # 300
    parser.add_argument("--patience",             type=int,       default=20,       help="Patience for early stopping")
    parser.add_argument("--batch_size",           type=int,       default=32,       help="Batch size")
    parser.add_argument("--lr",                   type=float,     default=1e-3,     help="Learning rate")
    parser.add_argument("--seed",                 type=int,       default=0)
    parser.add_argument("--checkpoint_path",      type=str,       default="src/forecasting/checkpoint.pt",       help="Path to save checkpoint")
    # visualization
    parser.add_argument("--n_plots",              type=int,       default=32,                                    help="Number of plots")
    parser.add_argument("--save_plots_path",      type=str,       default="results/forecasting/contam",          help="Path to save plots") # dataset parameter
    parser.add_argument("--results_file",         type=str,       default="results/results.txt",                 help="Path to file to save results in")
    return parser.parse_args()


def get_data_loaders(args):
    N_input = int(args.sequence_split*args.timesteps)  # input length
    N_output = args.timesteps - N_input                # target length

    train_data = F_Dataset(args.train_dataset_path, ts_split=args.sequence_split)
    test_data = F_Dataset(args.test_dataset_path, ts_split=args.sequence_split, return_date=True)

    trainloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
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
    return trainloader, testloader, N_input, N_output


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get data loaders
    trainloader, testloader, N_input, N_output = get_data_loaders(args)
    
    # build model
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

    # train model
    train_loss_evol, smape_loss, mae_loss, mse_loss, rmse_loss, mape_loss, mase_loss, r2_loss = train_model(
        trainloader, testloader,
        model, learning_rate=args.lr,
        epochs=args.epochs, patience=args.patience,
        checkpoint_path = args.checkpoint_path,
        device=device,
        verbose=1,
    )

    # save results
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    print(
        f"train_dataset_path: {args.train_dataset_path}\n\
        Final: smape={smape_loss}, mae={mae_loss}, mse={mse_loss}, rmse={rmse_loss}, mape={mape_loss}, mase={mase_loss}, r2={r2_loss}",
        file=open(args.results_file, "a")
    )
    plt.plot(train_loss_evol)
    os.makedirs(args.save_plots_path, exist_ok=True)
    plt.savefig(args.save_plots_path + "/forecast_train_loss_evol.jpg")

    # plot predictions
    if args.n_plots:
        count = 0
        plt.figure(figsize=(10, 6))
        for dates, inputs, targets in testloader:
            dates = np.array(dates)
            inputs  = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs).to(device)
            for i in range(args.batch_size):
                # if count == args.n_plots:
                #     return smape_loss, mae_loss, mse_loss, rmse_loss, mape_loss, mase_loss, r2_loss
                input = inputs.detach().cpu().numpy()[i,:,:]
                target = targets.detach().cpu().numpy()[i,:,:]
                pred = preds.detach().cpu().numpy()[i,:,:]
                plt.plot(range(0, N_input), input, label='Model Input', linewidth=3)
                plt.plot(range(N_input-1, N_input+N_output), np.concatenate([input[N_input-1:N_input], target]), label='Target (GT)', linewidth=3)   
                plt.plot(range(N_input-1, N_input+N_output),  np.concatenate([input[N_input-1:N_input], pred]), label='Prediction', linewidth=3)       
                
                date_list = dates[:,i]
                next_date = str(pd.to_datetime(date_list[-1]) + pd.Timedelta(days=1)).split(' ')[0]
                date_list = [str(d).split(' ')[0] for d in date_list] + [next_date]
                plt.xticks(range(0, N_input+N_output+1, N_output), date_list)
                plt.xlabel('Date')
                plt.ylabel('Load (normalized)')
                plt.title('Load Forecasting')
                plt.legend()
                plt.savefig(f"{args.save_plots_path}/{count}.jpg")
                plt.clf()
                count += 1

    return smape_loss, mae_loss, mse_loss, rmse_loss, mape_loss, mase_loss, r2_loss


if __name__ == "__main__":
    args = parse_args()
    run(args)

# TODO park office and inpg datasets `might` need series clustering first
