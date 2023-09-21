import argparse
import os

import matplotlib.pyplot as plt
import torch
from dataset import TS_Dataset
from feature_extractor import LSTM_AE

# ---
# Parameters
# ---
parser = argparse.ArgumentParser(description="Train the feature extraction model and save its weights")

parser.add_argument("--dataset_root", default="data/inpg_dataset/npy_data/", help="Path to dataset root")
parser.add_argument("--checkpoint_path", default="anomaly-detection/checkpoint.pt", help="Path to checkpoint")
parser.add_argument("--path_to_save_plot", default="results/out_figs", help="Path to save plots")
parser.add_argument("--seq_len", type=int, default=24*3, help="Sequence length")
parser.add_argument("--nbr_variables", type=int, default=1, help="Number of variables")
parser.add_argument("--embedding_dim", type=int, default=24*3, help="Embedding dimension")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--every_epoch_print", type=int, default=1, help="Print every epoch")
parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
parser.add_argument("--max_grad_norm", type=float, default=0.05, help="Maximum gradient norm")
parser.add_argument("--results_file", default="results/results.txt", help="Path to file to save results in")

args = parser.parse_args()

# ---
# Load data
# ---
train_dataset = TS_Dataset(args.dataset_root, "train")# both train and test datasets contain anomalies
test_dataset = TS_Dataset(args.dataset_root, "test")

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True,
) 
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True,
)

# ---
# Train model
# ---
model = LSTM_AE(
    args.seq_len, 
    args.nbr_variables, 
    args.embedding_dim, 
    args.learning_rate, 
    args.every_epoch_print, 
    args.epochs, args.patience, 
    args.max_grad_norm, 
    checkpoint_path=args.checkpoint_path
)
loss_history = model.fit(train_dataloader)
plt.plot(loss_history)
os.makedirs(os.path.join(args.path_to_save_plot, "reconstruction"), exist_ok=True)
plt.savefig(args.path_to_save_plot+"/fe_loss_evol.png")
print(f"\nFinal reconstruction loss: {loss_history[-1]}", file=open(args.results_file, "a"))

# ---
# Visualize reconstruction
# ---
loaded_model = LSTM_AE(
    args.seq_len,
    args.nbr_variables, 
    args.embedding_dim,
    args.learning_rate, 
    args.every_epoch_print, 
    args.epochs, 
    args.patience, 
    args.max_grad_norm
)
loaded_model.load(args.checkpoint_path)

k = 0
flag = 0
for batch_idx, batch in enumerate(test_dataloader):
    batch_data = batch["data"]
    gt = batch["is_anomaly"]
    encoded_batch, decoded_batch = loaded_model(batch_data)
    for data_sample, decoded_sample, is_anomaly in zip(batch_data, decoded_batch, gt):
        if not is_anomaly: continue
        plt.clf()
        plt.title(f"is_anomaly: {is_anomaly}")
        plt.plot(data_sample.squeeze().cpu().data, label="input")
        plt.plot(decoded_sample.cpu().data, label="output")
        plt.legend()
        plt.savefig(os.path.join(args.path_to_save_plot, "reconstruction") + f"/{k}.png")
        k += 1
        if k>=10: 
            flag = 1
            break
    if flag: break
