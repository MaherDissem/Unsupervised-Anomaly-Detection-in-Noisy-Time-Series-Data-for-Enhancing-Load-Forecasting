import numpy as np
import torch
from src.loss.dilate_loss import dilate_loss
from tslearn.metrics import dtw, dtw_path


def train_model(
        trainloader, testloader,
        net, loss_type, learning_rate, epochs=1000, gamma=0.001, Lambda=1, alpha=0.5, 
        print_every=1, eval_every=5, verbose=1,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        log_file = "results/results.txt"
    ):

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    mse_criterion = torch.nn.MSELoss()
    losses = []
    for epoch in range(1, epochs): 
        for i, data in enumerate(trainloader, 0):
            inputs, target = data
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)
            batch_size, N_output = target.shape[0:2]
            # forward + backward + optimize
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
            losses.append(loss.item())
        if verbose:
            if epoch % print_every == 0:
                print('epoch ', epoch, ' loss ', loss.item(),' loss shape ', loss_shape.item(), ' loss temporal ', loss_temporal.item())
            if epoch % eval_every == 0:
                eval_model(net, testloader, gamma, device, verbose=1)
    
    mse_loss, mae_loss, mape_loss, dtw_loss, tdi_loss = eval_model(net, testloader, gamma, device, verbose=0)
    print(f" mse_loss: {mse_loss}, mae_loss: {mae_loss}, mape_loss: {mape_loss}, dtw_loss: {dtw_loss}, tdi_loss: {tdi_loss}",
          file=open(log_file, "a"))
    return losses
  

def eval_model(net, loader, gamma, device, verbose=1):   
    criterion = torch.nn.MSELoss()
    losses_mse = []
    losses_mae = []
    losses_mape = []
    losses_dtw = []
    losses_tdi = []   
    for i, data in enumerate(loader, 0):
        loss_mse, loss_dtw, loss_tdi = torch.tensor(0),torch.tensor(0),torch.tensor(0)
        # get the inputs
        inputs, target = data
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)
        batch_size, N_output = target.shape[0:2]
        outputs = net(inputs)
        # MSE    
        loss_mse = criterion(target, outputs)  
        # MAE
        loss_mae = torch.mean(torch.abs(outputs - target))  
        # MAPE
        loss_mape = torch.mean(torch.abs((outputs - target) / target)) * 100
        # DTW and TDI
        loss_dtw, loss_tdi = 0, 0
        for k in range(batch_size):         
            target_k_cpu = target[k, :, 0:1].view(-1).detach().cpu().numpy()
            output_k_cpu = outputs[k, :, 0:1].view(-1).detach().cpu().numpy()
            path, sim = dtw_path(target_k_cpu, output_k_cpu)   
            loss_dtw += sim
            Dist = 0
            for i, j in path:
                    Dist += (i-j)*(i-j)
            loss_tdi += Dist / (N_output*N_output)                            
        loss_dtw = loss_dtw /batch_size
        loss_tdi = loss_tdi / batch_size

        losses_mse.append( loss_mse.item() )
        losses_mae.append( loss_mae.item() )
        losses_mape.append( loss_mape.item() )
        losses_dtw.append( loss_dtw )
        losses_tdi.append( loss_tdi )
    mse_loss = np.array(losses_mse).mean()
    mae_loss = np.array(losses_mae).mean()
    mape_loss = np.array(losses_mape).mean()
    dtw_loss = np.array(losses_dtw).mean()
    tdi_loss = np.array(losses_tdi).mean()
    if verbose:
        print( 'Eval mse=', mse_loss, 'mae=', mae_loss, 'mape=', loss_mape, ' dtw=', dtw_loss ,' tdi=', tdi_loss) 
    return mse_loss, mae_loss, mape_loss, dtw_loss, tdi_loss
