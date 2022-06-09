#%% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
# %% data import
iris = load_iris()
X = iris.data
y = iris.target

# %% train test split


# %% convert to float32

# %% dataset



# %% dataloader

# %% check dims

# %% define class

# %% hyper parameters
# NUM_FEATURES = ...
# HIDDEN = ...
# NUM_CLASSES = ...
# %% create model instance

# %% loss function
criterion = nn.CrossEntropyLoss()
# %% optimizer

# %% training
     
# %% show losses over epochs


# %% test the model


# %% Accuracy

