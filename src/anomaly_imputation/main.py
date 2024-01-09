import sys
sys.path.append("./src")
import os
import argparse
import torch
import matplotlib.pyplot as plt

from dataset_ai import AI_Dataset
from autoencoder import LSTM_AE
from utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Define hyperparameters for training")
    # data params
    parser.add_argument("--dataset_root",        type=str,   default="dataset/processed/Park/Office/30_minutes/ai_train/data",    help="Root directory of the dataset")
    parser.add_argument("--split_ratio",         type=float, default=0.8,                                           help="Ratio for train-test split")
    parser.add_argument("--seq_len",             type=int,   default=48*1,                                          help="Sequence length")
    parser.add_argument("--no_features",         type=int,   default=1,                                             help="Number of features")
    # model params
    parser.add_argument("--mask_size",           type=int,   default=8,                                             help="Length of the mask")
    parser.add_argument("--embedding_dim",       type=int,   default=128,                                           help="Dimension of embedding")
    parser.add_argument("--learning_rate",       type=float, default=1e-3,                                          help="Learning rate for the optimizer")
    parser.add_argument("--batch_size",          type=float, default=32,                                            help="Batch size for training")
    parser.add_argument("--seed",                type=int,   default=0,                                             help="Random seed")
    parser.add_argument("--checkpoint_path",     type=str,   default="src/anomaly_imputation/checkpoint.pt",        help="Path to save checkpoint")
    # training params
    parser.add_argument("--epochs",              type=int,   default=500,                                           help="Number of training epochs")
    parser.add_argument("--patience",            type=int,   default=50,                                            help="Patience for early stopping")
    parser.add_argument("--max_grad_norm",       type=float, default=0.05,                                          help="Maximum gradient norm for gradient clipping")
    # logging params
    parser.add_argument("--every_epoch_print",   type=int,   default=10,                                            help="Print results every n epochs")
    parser.add_argument("--save_eval_plots",     type=bool,  default=True,                                          help="Save evaluation plots")
    parser.add_argument("--save_folder",         type=str,   default="results/ai_eval_plots",                       help="Folder to save evaluation plots")
    return parser.parse_args()

def get_data_loaders(dataset_root, split_ratio, mask_size, batch_size):
    dataset = AI_Dataset(dataset_root,
                         mask_size)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(split_ratio * len(dataset)), len(dataset) - int(split_ratio * len(dataset))])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader

def train(args):
    train_dataloader, test_dataloader = get_data_loaders(args.dataset_root, args.split_ratio, args.mask_size, args.batch_size)
    model = LSTM_AE(args.seq_len, args.no_features, args.embedding_dim, args.learning_rate, args.every_epoch_print, args.epochs, args.patience, args.max_grad_norm, args.checkpoint_path, args.seed)
    loss_history = model.fit(train_dataloader, test_dataloader)

    if args.save_eval_plots:
        plt.plot(loss_history)
        os.makedirs(args.save_folder, exist_ok=True)
        plt.savefig(os.path.join(args.save_folder, "loss_history.png"))
        plt.clf()

        del model
        loaded_model = LSTM_AE(args.seq_len, args.no_features, args.embedding_dim, args.learning_rate, args.every_epoch_print, args.epochs, args.patience, args.max_grad_norm, args.checkpoint_path, args.seed)
        loaded_model.load()

        i = 0
        for batch in test_dataloader:
            ts_batch = batch["masked_data"] # torch.Size([32, 48, 1])
            mask_batch = batch["mask"]
            gt_ts_batch = batch["clean_data"]
            model_out_batch = loaded_model.infer(ts_batch)

            for ts, mask, gt_ts, model_out in zip(ts_batch, mask_batch, gt_ts_batch, model_out_batch):
                mask = mask.squeeze(0).squeeze(-1).detach().cpu()
                gt_ts = gt_ts.squeeze(0).squeeze(-1).detach().cpu()
                filled_ts = ts.clone().squeeze(0).squeeze(-1).detach().cpu()
                model_out = model_out.squeeze(0).squeeze(-1).detach().cpu()
                filled_ts[mask==0] = model_out[mask==0]

                plt.plot(ts.squeeze(0).squeeze(-1), label="serie with missing values")
                plt.plot(gt_ts, label="GT: ground truth")
                plt.plot(model_out, label="autoencoder's output")
                plt.plot(filled_ts.squeeze(0).squeeze(-1), label="serie with filled values")
                plt.legend()
                plt.savefig(os.path.join(args.save_folder, f"{i}.png"))
                plt.clf()
                i += 1


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(args)

# TODO: compare autoencoder vs KNN 