import sys
sys.path.append("./src") # TODO: fix this hack

import torch
import matplotlib.pyplot as plt

from dataset import TS_Dataset
from model import LSTM_AE
from utils.utils import set_seed

dataset_root = "../../dataset/processed/AEMO/test/ad_train_contam"
split_ratio = 0.8
seq_len = 48
no_features = 1
embedding_dim = 128
learning_rate = 1e-3
every_epoch_print = 1
epochs = 200
patience = 20
max_grad_norm = 0.05


set_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

dataset = TS_Dataset(dataset_root)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(split_ratio * len(dataset)), len(dataset) - int(split_ratio * len(dataset))])

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=True,
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
)

model = LSTM_AE(seq_len, no_features, embedding_dim, learning_rate, every_epoch_print, epochs, patience, max_grad_norm)
loss_history = model.fit(train_dataloader)

# Evaluate
plt.plot(loss_history)
del model
loaded_model = LSTM_AE(seq_len, no_features, embedding_dim, learning_rate, every_epoch_print, epochs, patience, max_grad_norm)
loaded_model.load()

for i, batch in enumerate(test_dataloader):
    ts = batch["masked_data"]
    mask = batch["mask"]
    gt_ts = batch["clean_data"]

    model_out = loaded_model.infer(ts)
    model_out = model_out.squeeze(0).squeeze(-1).detach().cpu()
    filled_ts = ts.clone().squeeze(0).squeeze(-1).detach().cpu()
    mask = mask.squeeze(0).squeeze(-1).detach().cpu()
    filled_ts[mask==0] = model_out[mask==0]
    gt_ts = gt_ts.squeeze(0).squeeze(-1).detach().cpu()

    plt.plot(gt_ts, label="GT: ground truth")
    plt.plot(ts.squeeze(0).squeeze(-1), label="serie with missing values")
    plt.plot(model_out, label="autoencoder's output")
    plt.plot(filled_ts.squeeze(0).squeeze(-1), label="serie with filled values")
    plt.legend()
    plt.show()
    plt.clf()
    if i > 10: break
