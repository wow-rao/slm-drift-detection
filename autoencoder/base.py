import torch.nn as nn
from abc import ABC, abstractmethod


class BaseAutoencoder(ABC, nn.Module):
    
    def __init__(self):
        super(BaseAutoencoder, self).__init__()
    
    @abstractmethod
    def encode(self, x):
        pass
    
    @abstractmethod
    def decode(self, x):
        pass
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
