import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def multilayer_perceptron(block_size, emb_size, n_units):
    # Generating params
    # Embedding layer
    g = torch.Generator().manual_seed(42)
    C = torch.randn((27, emb_size), generator=g).to(device)
    
    # First hidden layer
    W1 = torch.randn((block_size * emb_size, n_units), generator=g).to(device)
    b1 = torch.randn((n_units,), generator=g).to(device)
    
    # Output layer
    W2 = torch.randn((n_units, 27), generator=g).to(device)
    b2 = torch.randn(27, generator=g).to(device)
    
    params = [C, W1, b1, W2, b2]
    
    return params

def forward(params, X, block_size=3):
    C, W1, b1, W2, b2 = params
    emb = C[X]
    h = torch.tanh(emb.view(-1, block_size * C.shape[1])@W1 + b1)
    logits = h@W2 + b2
    return logits

class MLP():
    """
    The following class aims to do a Multilayer Perceptron, using 
    the sequential class from PyTorch. It should be fairly similar 
    to the multilayer_perceptron function, but allowing several 
    layers. 
    """
    def __init__(self, block_size : int = 3, 
                 emb_size : int = 2, 
                 n_units : int = 100, 
                 n_layers : int = 1):
        self.model = nn.Sequential().to(device)
        self.model.add_module('embedding', nn.Embedding(27, emb_size))
        for i in range(n_layers):
            self.model.add_module(f'linear_{i}', nn.Linear(block_size * emb_size, n_units))
            self.model.add_module(f'tanh_{i}', nn.Tanh())
        self.model.add_module('output', nn.Linear(n_units, 27))
        self.parameters = list(self.model.parameters())
        
        
    def forward(self, X):
        return self.model(X)