import os
import numpy as np
import torch
import torch.nn as nn

from .SCINet import SCINet
from .SCINet_decompose import SCINet_decompose

import sys
sys.path.insert(0, os.getcwd())
from src.utils.early_stop import EarlyStopping


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj==1:
        lr_adjust = {epoch: args.lr * (0.95 ** (epoch // 1))}

    elif args.lradj==2:
        lr_adjust = {
            0: 0.0001, 5: 0.0005, 10:0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    else:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
    return lr


class ModelWrapper():

    def __init__(self, args):
        self.args = args
        self.input_dim = self.args.nbr_var
        self.timesteps = self.args.timesteps
        self.sequence_split = self.args.sequence_split
        self.input_len = int(self.timesteps * self.sequence_split)
        self.horizon = self.timesteps - self.input_len

        self.criterion = smooth_l1_loss if self.args.L1Loss else nn.MSELoss(size_average=False).cuda()
        self.device  = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.model = self._build_model().cuda()
    

    def _build_model(self):
        if self.args.decompose:
            model = SCINet_decompose(
                output_len=self.horizon,
                input_len=self.input_len,
                input_dim=self.input_dim,
                hid_size=self.args.hidden_size,
                num_stacks=self.args.stacks,
                num_levels=self.args.levels,
                num_decoder_layer=self.args.num_decoder_layer,
                concat_len=self.args.concat_len,
                groups=self.args.groups,
                kernel=self.args.kernel,
                dropout=self.args.dropout,
                single_step_output_One=self.args.single_step_output_One,
                positionalE=self.args.positionalEcoding,
                modified=True,
                RIN=self.args.RIN
            )
        else:
            model = SCINet(
                output_len=self.horizon,
                input_len=self.input_len,
                input_dim=self.input_dim,
                hid_size=self.args.hidden_size,
                num_stacks=self.args.stacks,
                num_levels=self.args.levels,
                num_decoder_layer=self.args.num_decoder_layer,
                concat_len=self.args.concat_len,
                groups=self.args.groups,
                kernel=self.args.kernel,
                dropout=self.args.dropout,
                single_step_output_One=self.args.single_step_output_One,
                positionalE=self.args.positionalEcoding,
                modified=True,
                RIN=self.args.RIN
            )
        # print(model)
        return model


    def _select_optimizer(self):
        return torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=1e-5)


    def train(self,
              trainloader, validloader, testloader,
        ):
        
        optim=self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=False, checkpoint_path=self.args.checkpoint_path)
        epoch_start = 0
        loss_evol = []
        
        for epoch in range(epoch_start, self.args.epochs):
            self.model.train()
            total_loss = 0.0
            epoch_loss = 0.0
            adjust_learning_rate(optim, epoch, self.args)

            for data in trainloader:
                inputs, targets = data
                inputs = inputs.to(self.device)  # [batch_size, seq_len, input_size=n_var]
                targets = targets.to(self.device) # [batch_size, horizon, input_size]

                # inference
                self.model.zero_grad()
                if self.args.stacks == 1:
                    forecast = self.model(inputs)
                    loss = self.criterion(forecast, targets)
                if self.args.stacks == 2:
                    forecast, mid = self.model(inputs)
                    loss = self.criterion(forecast, targets) + self.criterion(mid, targets)
                epoch_loss += loss.item()

                # backpropagate
                loss.backward()
                total_loss += loss.item()
                optim.step()

            epoch_loss /= len(trainloader) # average loss per batch
            loss_evol.append(epoch_loss)
            
            # valid
            smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = self.validate(validloader)
            
            if self.args.verbose:
                if epoch % self.args.eval_every == 0:
                    print(f"epoch: {epoch}, Train loss: {epoch_loss:.7f}")
                    print(f"Eval: smape={smape_loss:.7f}, mae={mae_loss:.7f}, mse={mse_loss:.7f}, rmse={rmse_loss:.7f}, r2={r2_loss:.7f}")

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(mse_loss, self.model)

            if early_stopping.early_stop:
                break

        # load the last checkpoint with the best model (saved by EarlyStopping)
        self.model.load_state_dict(torch.load(self.args.checkpoint_path))

        smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = self.validate(testloader)
        return loss_evol, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss


    def validate(self, dataloader):
        self.model.eval()

        losses_smape = []
        losses_mae = []
        losses_mse = []
        losses_rmse = []
        losses_r2 = []

        for _data in dataloader:
            _, inputs, targets = _data
            inputs = inputs.to(self.device)   # [batch_size, window_size, n_var]
            targets = targets.to(self.device) # [batch_size, horizon, n_var]
            with torch.no_grad():
                if self.args.stacks == 1:
                    outputs = self.model(inputs)
                elif self.args.stacks == 2:
                    outputs, _ = self.model(inputs)

            # sMAPE
            absolute_percentage_errors = 2 * torch.abs(outputs - targets) / (torch.abs(outputs) + torch.abs(targets))
            loss_smape = torch.mean(absolute_percentage_errors) * 100
            # MAE
            loss_mae = torch.mean(torch.abs(outputs - targets))
            # MSE
            loss_mse = torch.mean((outputs - targets)**2)
            # RMSE
            loss_rmse = torch.sqrt(loss_mse)
            # R squared
            loss_r2 = 1 - torch.sum((targets - outputs)**2) / torch.sum((targets - torch.mean(targets))**2)

            losses_smape.append(loss_smape.item())
            losses_mae.append(loss_mae.item())
            losses_mse.append(loss_mse.item())
            losses_rmse.append(loss_rmse.item())
            losses_r2.append(loss_r2.item())

        smape_loss = np.array(losses_smape).mean()
        mae_loss = np.array(losses_mae).mean()
        mse_loss = np.array(losses_mse).mean()
        rmse_loss = np.array(losses_rmse).mean()
        r2_loss = np.array(losses_r2).mean()
        
        return smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss
    

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            if self.args.stacks == 1:
                outputs = self.model(inputs)
            elif self.args.stacks == 2:
                outputs, _ = self.model(inputs)
        return outputs
