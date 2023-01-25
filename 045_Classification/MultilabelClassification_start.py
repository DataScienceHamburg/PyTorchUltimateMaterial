#%% packages
from ast import Mult
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns
import numpy as np
from collections import Counter
# %% data prep
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size = 0.2)


# %% dataset and dataloader
class MultilabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# TODO: create instance of dataset

# TODO: create train loader


# %% model
# TODO: set up model class
# topology: fc1, relu, fc2
# final activation function??


# TODO: define input and output dim
# input_dim = ??
# output_dim = ??

# TODO: create a model instance


# %% loss function, optimizer, training loop
# TODO: set up loss function and optimizer
# loss_fn = ??
# optimizer = ??
losses = []
slope, bias = [], []
number_epochs = 100

# TODO: implement training loop
for epoch in range(number_epochs):
    pass
    # for j, data in enumerate(train_loader):
        
        # optimization


        # forward pass


        # compute loss

        
        # backward pass


        # update weights

        
    # TODO: print epoch and loss at end of every 10th epoch
    
    
# %% losses
# TODO: plot losses

# %% test the model
# TODO: predict on test set


#%% Naive classifier accuracy
# TODO: convert y_test tensor [1, 1, 0] to list of strings '[1. 1. 0.]'

# TODO: get most common class count

# TODO: print naive classifier accuracy


# %% Test accuracy
# TODO: get test set accuracy

# %%
