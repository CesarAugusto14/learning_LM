import torch
from models import MLP
from utils import tokenization, test_train_split, build_dataset
from time import time
block_size = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MLP(block_size=3, emb_size=2, n_units=100, n_layers=1)
# model = model.to(device)


print('Loading the data...')
words, chars, stringToIndex, indexToString = tokenization()
train_words, dev_words, test_words = test_train_split(words)
print('Done\n')

print('Building the dataset...\n')
X_train, Y_train = build_dataset(train_words, block_size, stringToIndex)
X_dev, Y_dev = build_dataset(dev_words, block_size, stringToIndex)
X_test, Y_test = build_dataset(test_words, block_size, stringToIndex)


# training
print('Training the model...')
t1 = time()
# model.train()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
for epoch in range(10000):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, Y_train)
    loss.backward()
    optimizer.step()
t2 = time()
print(f'Training time: {t2 - t1:.2f} seconds\n')
