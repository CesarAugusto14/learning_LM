import torch
from torch.nn import functional as F
from models import forward

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def tokenization(data = 'data/names.txt'):
    words = open(data, 'r').read().split()
    chars = sorted(list(set(''.join(words))))
    stringToIndex = {char:index + 1 for index, char in enumerate(chars)}
    stringToIndex['.'] = 0
    indexToString = {index:char for char, index in stringToIndex.items()}
    
    return words, chars, stringToIndex, indexToString

def build_dataset(words, block_size, stringToIndex):
    X, Y = [], []
    for word in words:
        context = [0] * block_size
        for ch in word + '.':
            ix = stringToIndex[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # Crop and append
    return torch.tensor(X).to(device), torch.tensor(Y).to(device)

def test_train_split(words):
    n = len(words)
    train_words = words[:int(.8*n)]
    dev_words = words[int(.8*n):int(.9*n)]
    test_words = words[int(.9*n):]
    return train_words, dev_words, test_words

def train(params, X, Y, X_dev, Y_dev, n_epochs = 1000, learning_rate = .5, batch_size = 128):
    hist_loss = []
    for p in params:
        p.requires_grad_()
    
    g = torch.Generator().manual_seed(42)
    for _ in range(n_epochs):
        # Mini-batch construction:
        ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)

        # Forward pass:
        logits = forward(params, X[ix])
        loss = F.cross_entropy(logits, Y[ix])

        # Backward pass
        for p in params:
            p.grad = None
        loss.backward()
        
        # Update
        for p in params:
            p.data -= learning_rate * p.grad
        
        hist_loss.append(loss.item())
    
    # Print loss with the dev set:
    logits = forward(params, X_dev)
    dev_loss = F.cross_entropy(logits, Y_dev)
    print(dev_loss.item())
    
    return params, hist_loss