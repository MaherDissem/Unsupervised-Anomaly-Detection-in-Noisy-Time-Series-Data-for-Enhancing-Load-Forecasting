import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset_f import DatasetForecasting
from seq2seq.train import ModelWrapper as Seq2seq_model
from SCINet.experiment import ModelWrapper as SCINet_model

import sys
sys.path.insert(0, os.getcwd())
from src.utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Runs Load Forecasting experiments")
    # dataset
    parser.add_argument("--train_dataset_path",   type=str,       default="dataset/processed/AEMO/NSW/exp0/lf_contam",       help="Path to train dataset")
    parser.add_argument("--test_dataset_path",    type=str,       default="dataset/processed/AEMO/NSW/exp0/lf_cleaned",    help="Path to clean dataset for testing")
    # sequence
    parser.add_argument("--timesteps",            type=int,       default=48*6,     help="Number of timesteps")
    parser.add_argument("--sequence_split",       type=float,     default=5/6,      help="Ratio of input to target (forecasting horizon) split")
    parser.add_argument("--nbr_var",              type=int,       default=1,        help="Number of variables")
    # training
    parser.add_argument("--epochs",               type=int,       default=500,      help="Number of epochs")
    parser.add_argument("--patience",             type=int,       default=50,       help="Patience for early stopping")
    parser.add_argument("--batch_size",           type=int,       default=32,       help="Batch size")
    parser.add_argument("--lr",                   type=float,     default=1e-3,     help="Learning rate")
    parser.add_argument("--seed",                 type=int,       default=0)
    parser.add_argument("--checkpoint_path",      type=str,       default="src/forecasting/checkpoint.pt",       help="Path to save checkpoint")
    parser.add_argument("--verbose",              type=bool,      default=True,     help="Verbosity")
    parser.add_argument("--eval_every",            type=int,       default=10,       help="Evaluate every n epochs")
    # visualization
    parser.add_argument("--n_plots",              type=int,       default=32,                                    help="Number of plots")
    parser.add_argument("--save_plots_path",      type=str,       default="results/forecasting/contam",          help="Path to save plots")
    parser.add_argument("--results_file",         type=str,       default="results/results.txt",                 help="Path to file to save results in")
    # model selection
    parser.add_argument("--model_choice",         type=str,       default="scinet", help="Model to use for forecasting: seq2seq or scinet")
    # seq2seq2 model parameters, only relevant if model == "seq2seq"
    parser.add_argument("--hidden_size",          type=int,       default=128,      help="Hidden size of the model")
    parser.add_argument("--num_grulstm_layers",   type=int,       default=1,        help="Number of GRU/LSTM layers")
    parser.add_argument("--fc_units",             type=int,       default=16,       help="Number of fully connected units") 
    # SCINet model parameters, only relevant if model == "scinet"
    ### -------  training settings --------------  
    parser.add_argument('--save', type=str, default='model/model.pt', help='path to save the final model')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
    parser.add_argument('--lradj', type=int, default=2,help='adjust learning rate')
    parser.add_argument('--save_path', type=str, default='exp/financial_checkpoints/')
    parser.add_argument('--model_name', type=str, default='SCINet')
    ### -------  model settings --------------  
    parser.add_argument('--concat_len', type=int, default=0)
    parser.add_argument('--hidden-size', default=1.0, type=float, help='hidden channel of module')# H, EXPANSION RATE
    parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
    parser.add_argument('--kernel', default=5, type=int, help='kernel size')#k kernel size
    parser.add_argument('--dilation', default=1, type=int, help='dilation')
    parser.add_argument('--positionalEcoding', type = bool , default=False)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--levels', type=int, default=3) # 3 for 24*5 and 48*5, 48*7*2, 24*7*2 input length
    parser.add_argument('--num_decoder_layer', type=int, default=1)
    parser.add_argument('--stacks', type=int, default=1)
    parser.add_argument('--long_term_forecast', action='store_true', default=False)
    parser.add_argument('--RIN', type=bool, default=False)
    parser.add_argument('--decompose', type=bool,default=False)
    parser.add_argument('--single_step', type=int, default=0, help='only supervise the final setp')
    parser.add_argument('--single_step_output_One', type=int, default=0, help='only output the single final step')
    parser.add_argument('--lastWeight', type=float, default=0.5,help='Loss weight lambda on the final step')

    args = parser.parse_args()
    if not args.long_term_forecast:
        args.concat_len = args.timesteps * (args.sequence_split - (1 - args.sequence_split))

    return parser.parse_args()


def get_data_loaders(args):
    N_input = int(args.sequence_split*args.timesteps)  # model input length 
    N_output = args.timesteps - N_input                # target length (forecasting horizon)

    train_data = DatasetForecasting(args.train_dataset_path, ts_split=args.sequence_split)
    test_data = DatasetForecasting(args.test_dataset_path, ts_split=args.sequence_split, return_date=True)
    valid_data, test_data = torch.utils.data.random_split(test_data, [int(0.5*len(test_data)), len(test_data) - int(0.5*len(test_data))])

    trainloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    validloader = DataLoader(
        valid_data,
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
    return trainloader, validloader, testloader, N_input, N_output


def plot_predictions(model, testloader, args, device, N_input, N_output):
    count = 0
    plt.figure(figsize=(10, 6))
    for dates, inputs, targets in testloader:
        dates = np.array(dates)
        inputs  = inputs.to(device)
        targets = targets.to(device)
        preds = model.predict(inputs)
        for i in range(args.batch_size):
            # if count == args.n_plots:
            #     return
            input = inputs.detach().cpu().numpy()[i,:,:]
            target = targets.detach().cpu().numpy()[i,:,:]
            pred = preds.detach().cpu().numpy()[i,:,:]
            plt.plot(range(0, N_input), input, label='Model Input', linewidth=3)
            plt.plot(range(N_input-1, N_input+N_output), np.concatenate([input[N_input-1:N_input], target]), label='Target (GT)', linewidth=3)   
            plt.plot(range(N_input-1, N_input+N_output),  np.concatenate([input[N_input-1:N_input], pred]), label='Prediction', linewidth=3)       
            
            date_list = dates[:,i]
            next_date = str(pd.to_datetime(date_list[-1]) + pd.Timedelta(days=1)).split(' ')[0]
            date_list = [str(d).split(' ')[0] for d in date_list] + [next_date]
            plt.xticks(range(0, N_input+N_output+1, (N_input+N_output)//(len(date_list)-1)), date_list)
            plt.xlabel('Date')
            plt.ylabel('Load (normalized)')
            plt.title('Load Forecasting')
            plt.legend()
            plt.savefig(f"{args.save_plots_path}/{count}.jpg")
            plt.clf()
            count += 1


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get data loaders
    trainloader, validloader, testloader, N_input, N_output = get_data_loaders(args)
    
    # build model
    if args.model_choice == "seq2seq":
        model = Seq2seq_model(args, N_input, N_output)
    elif args.model_choice == "scinet":
        model = SCINet_model(args)

    # train model
    train_loss_evol, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = model.train(trainloader, validloader, testloader)

    # save results
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    print(
        f"train_dataset_path: {args.train_dataset_path}\n\
        Final test: smape={smape_loss}, mae={mae_loss}, mse={mse_loss}, rmse={rmse_loss}, r2={r2_loss}",
        file=open(args.results_file, "a")
    )
    plt.plot(train_loss_evol)
    os.makedirs(args.save_plots_path, exist_ok=True)
    plt.savefig(args.save_plots_path + "/forecast_train_loss_evol.jpg")

    # plot predictions
    if args.n_plots:
        plot_predictions(model, testloader, args, device, N_input, N_output)

    # return metrics to pipeline
    return smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss


if __name__ == "__main__":
    args = parse_args()
    run(args)
