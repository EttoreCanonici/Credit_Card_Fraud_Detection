import torch
import torch.nn as nn
class Autoencoder(nn.Module):
    """
    Classe dell'autoencoder, accetta come argomenti la dimensione di input e la probabilità di dropout
    """
    def __init__(self, input_dim, hidden_dim=6, dropout_prob=0):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(64, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(12, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(12, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(128, self.input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x