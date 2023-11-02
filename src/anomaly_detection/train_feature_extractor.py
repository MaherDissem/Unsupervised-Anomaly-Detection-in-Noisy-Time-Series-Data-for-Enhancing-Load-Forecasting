import argparse
import os

import matplotlib.pyplot as plt
import torch

from feature_extractor import LSTM_AE
from dataset import TS_Dataset

# ---
# Parameters
# ---
parser = argparse.ArgumentParser(description="Train the feature extraction model and save its weights")
# data parameters
parser.add_argument("--train_dataset", default="dataset/processed/INPG/ad_train_contam", help="Path to dataset root")
parser.add_argument("--test_dataset", default="dataset/processed/INPG/ad_test_contam", help="Path to dataset root")
parser.add_argument("--seq_len", type=int, default=24*3, help="Sequence length")
parser.add_argument("--nbr_variables", type=int, default=1, help="Number of variables")
# module output
parser.add_argument("--checkpoint_path", default="src/anomaly_detection/checkpoint.pt", help="Path to checkpoint")
parser.add_argument("--path_to_save_plot", default="results/out_figs", help="Path to save plots")
parser.add_argument("--n_plots_to_save", default=20, help="number of reconstruction plots to save")
parser.add_argument("--results_file", default="results/results.txt", help="Path to file to save results in")
# train parameters
parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--max_grad_norm", type=float, default=0.05, help="Maximum gradient norm")
parser.add_argument("--every_epoch_print", type=int, default=1, help="Print every epoch")
# model parameters
parser.add_argument("--embedding_dim", type=int, default=24*3, help="Embedding dimension")

args = parser.parse_args()

# ---
# Load data
# ---
train_dataset = TS_Dataset(args.train_dataset) # both train and test datasets contain anomalies
test_dataset = TS_Dataset(args.test_dataset)

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


# ---
# Visualize reconstruction
# --- # TODO make this optional
plt.plot(loss_history)
plt.title("Reconstruction loss evolution")
os.makedirs(os.path.join(args.path_to_save_plot, "reconstruction"), exist_ok=True)
plt.savefig(args.path_to_save_plot+"/fe_loss_evol.png")
print(f"\nFinal reconstruction loss: {loss_history[-1]}", file=open(args.results_file, "a"))

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
        # if is_anomaly: continue
        plt.clf()
        plt.title(f"is_anomaly: {is_anomaly}")
        plt.plot(data_sample.squeeze().cpu().data, label="input")
        plt.plot(decoded_sample.cpu().data, label="reconstructed")
        plt.legend()
        plt.savefig(os.path.join(args.path_to_save_plot, "reconstruction") + f"/{k}.png")
        k += 1
        if k>args.n_plots_to_save: 
            flag = 1
            break
    if flag: break
