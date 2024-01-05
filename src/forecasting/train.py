import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from src.utils.early_stop import EarlyStopping


def train_model(
        trainloader, validloader, testloader,
        net, learning_rate,
        epochs=1000, patience=20,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose=1, eval_every=5,
        checkpoint_path='src/forecasting/checkpoint.pt', 
    ):

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    mse_criterion = torch.nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, verbose=False, checkpoint_path=checkpoint_path)
    loss_evol = []
    for epoch in range(1, epochs):
        early_stopping.epoch = epoch 
        epoch_loss = 0.0

        for data in trainloader:
            inputs, target = data
            inputs = inputs.to(device)
            target = target.to(device)
            outputs = net(inputs)
            train_loss = mse_criterion(target, outputs)      
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()
        epoch_loss /= len(trainloader) # average loss per batch
        loss_evol.append(epoch_loss)

        smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = eval_model(net, validloader, device)

        if verbose:
            if epoch % eval_every == 0:
                print(f"epoch: {epoch}, Train loss: {epoch_loss:.7f}")
                print(f"Eval: smape={smape_loss:.7f}, mae={mae_loss:.7f}, mse={mse_loss:.7f}, rmse={rmse_loss:.7f}, r2={r2_loss:.7f}")

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(mse_loss, net)

        if early_stopping.early_stop:
            break
    
    # load the last checkpoint with the best model (saved by EarlyStopping)
    net.load_state_dict(torch.load(checkpoint_path))

    smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = eval_model(net, testloader, device)
    return loss_evol, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss
  

def eval_model(net, loader, device):   
    losses_smape = []
    losses_mae = []
    losses_mse = []
    losses_rmse = []
    losses_r2 = []

    for data in loader:
        # run inference
        _, inputs, target = data
        inputs = inputs.to(device)
        target = target.to(device)
        outputs = net(inputs)

        # sMAPE
        absolute_percentage_errors = 2 * torch.abs(outputs - target) / (torch.abs(outputs) + torch.abs(target))
        loss_smape = torch.mean(absolute_percentage_errors) * 100
        # MAE
        loss_mae = torch.mean(torch.abs(outputs - target))
        # MSE
        loss_mse = torch.mean((outputs - target)**2)
        # RMSE
        loss_rmse = torch.sqrt(loss_mse)
        # R squared
        loss_r2 = 1 - torch.sum((target - outputs)**2) / torch.sum((target - torch.mean(target))**2)

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

