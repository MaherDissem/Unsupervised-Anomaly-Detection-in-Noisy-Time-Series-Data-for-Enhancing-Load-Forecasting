import numpy as np
import torch
import torch.nn as nn


# ---
# Ensure reproductibility
# ---
def fix_seeds(seed, with_torch=True, with_cuda=True):
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
fix_seeds(0)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta= -0.00001, checkpoint_path='./checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print(f"Early Stopping activated. Final validation loss : {self.val_loss_min:.7f}")
                self.early_stop = True
        # if the current score does not exceed the best scroe, run the codes following below
        else:  
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss


# (1) Encoder
class Encoder(nn.Module):
    def __init__(self, seq_len, no_features, embedding_size):
        super().__init__()
        
        self.seq_len = seq_len
        self.no_features = no_features    # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size   # the number of features in the embedded points of the inputs' number of features
        self.hidden_size = (2 * embedding_size)  # The number of features in the hidden state h
        self.LSTM1 = nn.LSTM(
            input_size = no_features,
            hidden_size = embedding_size,
            num_layers = 1,
            batch_first=True
        )
        
    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        x, (hidden_state, cell_state) = self.LSTM1(x)  
        last_lstm_layer_hidden_state = hidden_state[-1,:,:]
        return last_lstm_layer_hidden_state
    
    
# (2) Decoder
class Decoder(nn.Module):
    def __init__(self, seq_len, no_features, output_size):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.hidden_size = (2 * no_features)
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size = no_features,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = True
        )

        self.fc = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
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
    def __init__(
            self,
            seq_len=240, 
            no_features=1,
            embedding_dim=128,
            learning_rate=1e-3,
            every_epoch_print=100,
            epochs=10000,
            patience=20,
            max_grad_norm=0.005,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            checkpoint_path='./checkpoint.pt'
        ):
        super().__init__()
        
        self.seq_len = seq_len
        self.no_features = no_features
        self.embedding_dim = embedding_dim
        self.device = device

        self.encoder = Encoder(self.seq_len, self.no_features, self.embedding_dim).to(self.device)
        self.decoder = Decoder(self.seq_len, self.embedding_dim, self.no_features).to(self.device)
        self.criterion = nn.MSELoss(reduction='mean')
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.every_epoch_print = every_epoch_print
        self.checkpoint_path = checkpoint_path
    
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
        early_stopping = EarlyStopping(patience=self.patience, verbose=False, checkpoint_path=self.checkpoint_path)
        loss_history = []

        for epoch in range(1 , self.epochs+1):
            # updating early_stopping's epoch
            early_stopping.epoch = epoch    
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                data = batch["data"].to(self.device)
                optimizer.zero_grad()
                encoded, decoded = self(data)
                running_loss = self.criterion(decoded , data)
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
        self.load_state_dict(torch.load(self.checkpoint_path))

        return loss_history
    
    def load(self, PATH='./checkpoint.pt'):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))


    def vectorize(self, data):
        """
            Extracts features from input timeseries: original sequence, the reconstruction residual (input-decoded) and the latent representation.
            input: [batch_size, nbr_timesteps, nbr_variables]
            output: [batch_size, 3, nbr_timesteps, nbr_variables]
        """
        self.eval()
        data = data.to(self.device)
        encoded, decoded = self(data)
        residual = data - decoded
        return torch.stack([data, encoded.unsqueeze(-1), residual]).permute(1,0,2,3)
