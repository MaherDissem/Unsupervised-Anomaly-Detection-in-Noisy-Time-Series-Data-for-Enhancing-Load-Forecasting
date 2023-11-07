import os
import sys
import numpy as np
import torch

from loss.dilate_loss import dilate_loss
from tslearn.metrics import dtw, dtw_path

sys.path.insert(0, os.getcwd())
from src.utils.early_stop import EarlyStopping


def train_model(
        trainloader, testloader,
        net, loss_type, learning_rate, gamma=0.001, Lambda=1, alpha=0.5, 
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

        for i, data in enumerate(trainloader):
            inputs, target = data
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)
            batch_size, N_output = target.shape[0:2]
            outputs = net(inputs)
            loss_mse, loss_shape, loss_temporal = torch.tensor(-1), torch.tensor(-1), torch.tensor(-1)
            if loss_type=='mse':
                loss_mse = mse_criterion(target, outputs)
                loss = loss_mse                    
            if loss_type=='dilate':    
                loss, loss_shape, loss_temporal = dilate_loss(outputs, target, alpha, gamma, device)             
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= (i+1)
        loss_evol.append(epoch_loss)

        if verbose:
            print(f"epoch: {epoch}, loss: {epoch_loss/(i+1)}")
            if epoch % eval_every == 0:
                smape_loss, mae_loss, mse_loss, rmse_loss, mape_loss, mase_loss, r2_loss = eval_model(net, testloader, gamma, device)
                print(f"Eval: smape={smape_loss}, mae={mae_loss}, mse={mse_loss}, rmse={rmse_loss}, mape={mape_loss}, mase={mase_loss}, r2={r2_loss}")

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(epoch_loss, net)

        if early_stopping.early_stop:
            break
    
    # load the last checkpoint with the best model (saved by EarlyStopping)
    net.load_state_dict(torch.load(checkpoint_path))

    smape_loss, mae_loss, mse_loss, rmse_loss, mape_loss, mase_loss, r2_loss = eval_model(net, testloader, gamma, device)
    return loss_evol, smape_loss, mae_loss, mse_loss, rmse_loss, mape_loss, mase_loss, r2_loss
  

def eval_model(net, loader, gamma, device):   
    losses_smape = []
    losses_mae = []
    losses_mse = []
    losses_rmse = []
    losses_mape = []
    losses_mase = []
    losses_dtw = []
    losses_tdi = []
    losses_r2 = []

    for i, data in enumerate(loader, 0):
        # run inference
        inputs, target = data
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)
        batch_size, N_output = target.shape[0:2]
        outputs = net(inputs)

        # eval metrics
        # sMAPE
        absolute_percentage_errors = 2 * torch.abs(outputs - target) / (torch.abs(outputs) + torch.abs(target))
        loss_smape = torch.mean(absolute_percentage_errors) * 100
        # MAE
        loss_mae = torch.mean(torch.abs(outputs - target))
        # MSE
        loss_mse = torch.mean((outputs - target)**2)
        # RMSE
        loss_rmse = torch.sqrt(loss_mse)
        # MAPE
        loss_mape = torch.mean(torch.abs(outputs - target) / (torch.abs(target)+1e-6)) # division by zero for null targets
        # MASE
        loss_mase = torch.mean(torch.abs(outputs - target) / loss_mae)
        # R squared
        loss_r2 = 1 - torch.sum((target - outputs)**2) / torch.sum((target - torch.mean(target))**2)
        # DTW and TDI
        # loss_dtw, loss_tdi = 0, 0
        # for k in range(batch_size):         
        #     target_k_cpu = target[k, :, 0:1].view(-1).detach().cpu().numpy()
        #     output_k_cpu = outputs[k, :, 0:1].view(-1).detach().cpu().numpy()
        #     path, sim = dtw_path(target_k_cpu, output_k_cpu)   
        #     loss_dtw += sim
        #     Dist = 0
        #     for i, j in path:
        #             Dist += (i-j)*(i-j)
        #     loss_tdi += Dist / (N_output*N_output)                            
        # loss_dtw = loss_dtw /batch_size
        # loss_tdi = loss_tdi / batch_size

        losses_smape.append( loss_smape.item() )
        losses_mae.append( loss_mae.item() )
        losses_mse.append( loss_mse.item() )
        losses_rmse.append( loss_rmse.item() )
        losses_mape.append( loss_mape.item() )
        losses_mase.append( loss_mase.item() )
        losses_r2.append( loss_r2.item() )
        # losses_dtw.append( loss_dtw )
        # losses_tdi.append( loss_tdi )

    smape_loss = np.array(losses_smape).mean()
    mae_loss = np.array(losses_mae).mean()
    mse_loss = np.array(losses_mse).mean()
    rmse_loss = np.array(losses_rmse).mean()
    mape_loss = np.array(losses_mape).mean()
    mase_loss = np.array(losses_mase).mean()
    r2_loss = np.array(losses_r2).mean()
    # dtw_loss = np.array(losses_dtw).mean()
    # tdi_loss = np.array(losses_tdi).mean()
    return smape_loss, mae_loss, mse_loss, rmse_loss, mape_loss, mase_loss, r2_loss #, dtw_loss, tdi_loss