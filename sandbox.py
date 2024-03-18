import torch 
from time import time
from models import multilayer_perceptron, forward
from utils import tokenization, build_dataset, test_train_split, train

# Setting up the device:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'The device is {device}\n')
block_size = 3

print('Loading the data...')
words, chars, stringToIndex, indexToString = tokenization()
train_words, dev_words, test_words = test_train_split(words)
print('Done\n')

print('Building the dataset...\n')
X_train, Y_train = build_dataset(train_words, block_size, stringToIndex)
X_dev, Y_dev = build_dataset(dev_words, block_size, stringToIndex)
X_test, Y_test = build_dataset(test_words, block_size, stringToIndex)

# Shapes of the dataset:
print(f'X_train shape: {X_train.shape} Y_train shape: {Y_train.shape}')
print(f'X_dev shape: {X_dev.shape} Y_dev shape: {Y_dev.shape}')
print(f'X_test shape: {X_test.shape} Y_test shape: {Y_test.shape}\n')

# Loading the model:
print('Loading the model...')
params = multilayer_perceptron(block_size, 3, 100)
print("number of parameters: " + str(sum(p.numel() for p in params)))

# Training the model:
print('Training the model...')
t1 = time()
params, hist_loss = train(params, 
                          X_train, Y_train, 
                          X_dev, Y_dev, 
                          n_epochs = 10000, 
                          learning_rate = .5, 
                          batch_size = X_train.shape[0])
t2 = time()
print(f'Training time: {t2 - t1:.2f} seconds\n')
# Save params
torch.save(params, 'params.pt')
