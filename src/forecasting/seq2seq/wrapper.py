import os
import numpy as np
import torch

from .model import DecoderRNN, EncoderRNN, Net_GRU

import sys
sys.path.insert(0, os.getcwd())
from src.utils.early_stop import EarlyStopping


class ModelWrapper():

    def __init__(self, args, N_input, N_output):
        self.args = args
        self.N_output = N_output
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()


    def build_model(self):
        encoder = EncoderRNN(
            input_size=self.args.nbr_var, 
            hidden_size=self.args.hidden_size, 
            num_grulstm_layers=self.args.num_grulstm_layers, 
            batch_size=self.args.batch_size
        ).to(self.device)

        decoder = DecoderRNN(
            input_size=self.args.nbr_var, 
            hidden_size=self.args.hidden_size, 
            num_grulstm_layers=self.args.num_grulstm_layers, 
            fc_units=self.args.fc_units, 
            output_size=self.args.nbr_var
        ).to(self.device)

        model = Net_GRU(encoder, decoder, target_length=self.N_output, device=self.device).to(self.device)
        return model

    def train(
            self,
            trainloader, validloader, testloader,
        ):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        mse_criterion = torch.nn.MSELoss()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=False, checkpoint_path=self.args.checkpoint_path)
        loss_evol = []
        for epoch in range(1, self.args.epochs):
            early_stopping.epoch = epoch
            epoch_loss = 0.0

            for data in trainloader:
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                train_loss = mse_criterion(targets[:,:,0], outputs[:,:,0])      
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item()
            epoch_loss /= len(trainloader) # average loss per batch
            loss_evol.append(epoch_loss)

            smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = self.eval_model(validloader)

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

        smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = self.eval_model(testloader)
        return loss_evol, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss
    

    def eval_model(self, loader):       
        losses_smape = []
        losses_mae = []
        losses_mse = []
        losses_rmse = []
        losses_r2 = []

        for data in loader:
            # run inference
            _, inputs, targets = data
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
            
            outputs = outputs[:,:,0]
            targets = targets[:,:,0]

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
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs).to(self.device)
            outputs = self.model(inputs)
        return outputs
