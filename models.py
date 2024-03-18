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

