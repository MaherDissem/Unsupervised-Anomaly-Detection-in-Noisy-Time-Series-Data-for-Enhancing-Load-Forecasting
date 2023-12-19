import torch
import torch.nn as nn
from utils.early_stop import EarlyStopping
from utils.utils import set_seed

set_seed(0) # TODO move to parent caller

# (1) Encoder
class Encoder(nn.Module):
    def __init__(self, seq_len, no_features, embedding_size):
        super().__init__()
        
        self.seq_len = seq_len
        self.no_features = no_features     # The number of expected features(= nbr of dimensions) in the input sequence x
        self.hidden_size = embedding_size  # The number of features in the hidden state h (embedding_size)
        self.num_layers = 1
        self.bidirectional = True
        
        self.biLSTM = nn.LSTM(
            input_size = no_features,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        
    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        x, (hidden_state, cell_state) = self.biLSTM(x)  
        # last_lstm_layer_hidden_state = hidden_state[-1,:,:] # for multi-layered LSTM
        return torch.mean(x, dim=1)
    
    
# (2) Decoder
class Decoder(nn.Module):
    def __init__(self, seq_len, no_features, output_size):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.hidden_size = (2 * no_features)
        self.output_size = output_size
        self.bidirectional = True
        self.num_layers = 1
        dir = 2 if self.bidirectional else 1

        self.LSTM1 = nn.LSTM(
            input_size = no_features*dir,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            bidirectional=self.bidirectional,
            batch_first = True,
        )
        self.fc = nn.Linear(self.hidden_size*dir, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1) # repeats latent vector seq_len times
        x, (hidden_state, cell_state) = self.LSTM1(x)
        out = self.fc(x)
        return out
    
# (3) Autoencoder : putting the encoder and decoder together
class LSTM_AE(nn.Module):
    """
    Parameters
        input_sequences: A list (or tensor) of shape [num_seqs, seq_len, num_features] representing your training set of sequences.
            Each sequence should have the same length, seq_len, and contain a sequence of vectors of size num_features.
            If num_features=1, then you can input a list of shape [num_seqs, seq_len] instead.
            [Notice] Currently TorchCoder can take [num_seqs, seq_len] as an input. Soon to be fixed.
        embedding_dim: Size of the vector encodings you want to create.
        learning_rate: Learning rate for the autoencoder. default = 1e-3
        every_epoch_print : Deciding the size of N to print the loss every N epochs. default = 100
        epochs: Total number of epochs to train for. default = 10000
        patience : Number of epochs to wait for if the loss does not decrease. default = 20
        max_grad_norm : Maximum size for gradient used in gradient descent (gradient clipping). default = 0.005

    Returns
        encoded: The encoded vector (representation) of the input_sequences.
        decoded: The decoded vector of encoded, which should be very close to the input_sequences.
        final_loss: The final mean squared error of the autoencoder on the training set.

    """
    def __init__(self, seq_len, no_features, embedding_dim, learning_rate=1e-3, every_epoch_print=100, epochs=10000, patience=20, max_grad_norm=0.005):
        super().__init__()
        
        self.seq_len = seq_len
        self.no_features = no_features
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(self.seq_len, self.no_features, self.embedding_dim).to(self.device)
        self.decoder = Decoder(self.seq_len, self.embedding_dim, self.no_features).to(self.device)
        self.criterion = nn.MSELoss(reduction='mean')
        self.is_fitted = False
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.every_epoch_print = every_epoch_print
        
    
    def forward(self, x):
        torch.manual_seed(0)
        encoded = self.encoder(x.to(self.device))
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def fit(self, train_loader):
        """
        trains the model's parameters over a fixed number of epochs, specified by `n_epochs`, as long as the loss keeps decreasing.
        :param dataset: `Dataset` object
        :param bool save: If true, dumps the trained model parameters as pickle file at `dload` directory
        :return:
        """
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        self.train()
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=False)
        loss_history = []

        for epoch in range(1 , self.epochs+1):
            # updating early_stopping's epoch
            early_stopping.epoch = epoch    
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                clean_ts = batch["clean_data"].to(self.device)
                masked_ts = batch["masked_data"].to(self.device)
                mask = batch["mask"].to(self.device)
                optimizer.zero_grad()
                encoded, decoded = self(masked_ts)

                # running_loss = self.criterion(clean_ts , decoded)
                # loss is difference between clean_ts and decoded for the masked_ts
                running_loss = self.criterion(clean_ts * mask, decoded * mask)
                
                epoch_loss += running_loss.item()
                # Backward pass
                running_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm) # clipping avoids exploding gradients
                optimizer.step()
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(epoch_loss, self)

            if early_stopping.early_stop:
                break
            
            if epoch % self.every_epoch_print == 0:
                print(f"epoch : {epoch}, loss_mean : {epoch_loss:.7f}")

            loss_history.append(epoch_loss)
        
        # load the last checkpoint with the best model (saved by EarlyStopping)
        self.load_state_dict(torch.load('./checkpoint.pt'))
        self.is_fitted = True

        return loss_history
    
    def load(self, PATH='./checkpoint.pt'):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))

    def infer(self, data):
        assert self.is_fitted, "Model is not fitted yet. Call fit() or load() before infer()"
        self.eval()
        data = data.to(self.device)
        encoded, decoded = self(data)
        return decoded
    
    def impute(self, ts, mask):
        assert self.is_fitted, "Model is not fitted yet. Call fit() or load() before infer()"
        self.eval()
        ts = ts.to(self.device)
        mask = mask.to(self.device)

        model_out = self.infer(ts)

        model_out = model_out.squeeze(0).squeeze(-1).detach().cpu()
        filled_ts = ts.clone().squeeze(0).squeeze(-1).detach().cpu()
        mask = mask.squeeze(0).squeeze(-1).detach().cpu()

        filled_ts[mask==0] = model_out[mask==0]
        return filled_ts