import sys
sys.path.append("./src") # TODO: fix this hack

import os
import torch

from dataset import TS_Dataset
from model import LSTM_AE
from utils.utils import set_seed
from train import parse_args

args = parse_args()
set_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loaded_model = LSTM_AE(args.seq_len, args.no_features, args.embedding_dim, args.learning_rate, args.every_epoch_print, args.epochs, args.patience, args.max_grad_norm)
loaded_model.load()

