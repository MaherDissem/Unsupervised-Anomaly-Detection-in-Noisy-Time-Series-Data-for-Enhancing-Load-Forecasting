import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

class DataLoaderH(object):
    def __init__(self, file_name, train_ratio, valid_ratio, horizon, window, normalize=2): # (self.args.data, 0.6, 0.2, self.args.horizon, self.args.window_size, self.args.normalize)
        self.P = window
        self.h = horizon
        # fin = open(file_name)
        # self.rawdat = np.loadtxt(fin, delimiter=',') # (52560, 1) == (nbr_time_steps, nbr_var)

        # load csv file instead of their shitty dataset
        csv_path = "dataset/processed/AEMO/NSW/exp1/load_contam.csv"
        df = pd.read_csv(csv_path)
        df = df["TOTALDEMAND"].values
        self.rawdat = df

        if self.rawdat.ndim == 1:
            self.rawdat = np.expand_dims(self.rawdat, axis=1) # unsqueeze for the case of 1d data
        self.dat = np.zeros(self.rawdat.shape) 

        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self.bias =  np.zeros(self.m)
        self._normalized(normalize)
        self._split(int(train_ratio * self.n), int((train_ratio + valid_ratio) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        self.bias = torch.from_numpy(self.bias).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.h, self.m)

        self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)
        self.bias = self.bias.cuda()
        self.bias = Variable(self.bias)

        tmp = tmp[:, -1, :].squeeze()
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))


    def _normalized(self, normalize):
        if normalize == 0: # no normalization
            self.dat = self.rawdat

        if normalize == 1: # normalized by the maximum value of entire matrix.
            self.dat = self.rawdat / np.max(self.rawdat)
        
        if normalize == 2: # normlized by the maximum value of each row (sensor).
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

        if normalize == 3: # normlized by the mean/std value of each row (sensor).
            for i in range(self.m):
                self.scale[i] = np.std(self.rawdat[:, i]) #std
                self.bias[i] = np.mean(self.rawdat[:, i])
                self.dat[:, i] = (self.rawdat[:, i] - self.bias[i]) / self.scale[i]
                
        if normalize == 4: # normlized by the mean/std value of each row (sensor).
            for i in range(self.m):
                self.scale[i] = np.std(self.rawdat[:int(self.dat.shape[0]*0.7), i]) #std
                self.bias[i] = np.mean(self.rawdat[:int(self.dat.shape[0]*0.7), i])
                self.dat[:, i] = (self.rawdat[:, i] - self.bias[i]) / self.scale[i]


    def _split(self, train, valid, test):
        # arguments are the start index of train/valid/text sets
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)

        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)


    def _batchify(self, idx_set, horizon):
        window_size = 6*48
        stride = 48 # TODO change to parameter
        horizon = 48

        n = len(idx_set)
        X = torch.zeros(((n-window_size)//stride+1, window_size, self.m))
        Y = torch.zeros(((n-window_size)//stride+1, horizon, self.m))
        
        start = 0
        end = start + window_size
        j = 0        
        while end < n:
            X[j, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[j, :, :] = torch.from_numpy(self.dat[end-horizon:end, :])
            start += stride
            end = start + window_size
            j += 1
        return [X, Y]


    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))

        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.cuda()
            Y = Y.cuda()
            yield Variable(X), Variable(Y) # (batch_size, seq_len, input_size), (batch_size, horizon, input_size)
            start_idx += batch_size
