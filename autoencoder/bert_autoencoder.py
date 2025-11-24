import torch.nn as nn
from .base import BaseAutoencoder


class BertAutoencoder(BaseAutoencoder):
    
    def __init__(self, input_dim=768, edim=256):
        super(BertAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, edim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(edim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim),
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
