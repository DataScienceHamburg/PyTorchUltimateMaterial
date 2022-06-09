#%% packages
from typing import OrderedDict
import torch
import torch.nn as nn

#%% sample input data of certain shape

# %%
model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(3, 8, 3)), # out: (BS, 8, 30, 30)
    ('relu1', nn.ReLU()),
    ('pool', nn.MaxPool2d(2, 2)), # out: (BS, 8, 15, 15)
    ('conv2', nn.Conv2d(8, 16, 3)), # out: (BS, 16, 13, 13)
    ('relu2', nn.ReLU()),
    ('pool2', nn.MaxPool2d(2, 2)), # out: (BS, 16, 6, 6)
    ('flatten', nn.Flatten()),  # shape: (3, 16*6*6)
    ('fc1', nn.Linear(16 * 6 * 6, 127)),
    ('relu3', nn.ReLU()),
    ('fc2', nn.Linear(128, 64)),
    ('relu4', nn.ReLU()),
    ('fc3', nn.Linear(64, 1)),
    ('sigmoid', nn.Sigmoid())
]))

# %% test the model setup
