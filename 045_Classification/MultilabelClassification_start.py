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

# create instance of dataset
multilabel_train_data = MultilabelDataset(X_train, y_train)
multilabel_test_data = MultilabelDataset(X_test, y_test)

# create train loader
train_loader = DataLoader(multilabel_train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(multilabel_test_data, batch_size=32, shuffle=True)


# %% model
# set up model class
# topology: fc1, relu, fc2
# final activation function??
class MultiLabelNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# define input and output dim
input_dim = multilabel_train_data.X.shape[1]
output_dim = multilabel_train_data.y.shape[1]

# create a model instance
model = MultiLabelNetwork(input_size=input_dim, hidden_size=20, output_size=output_dim)

# %% loss function, optimizer, training loop
# set up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
losses = []
slope, bias = [], []
number_epochs = 100

# implement training loop
for epoch in range(number_epochs):
    for j, (X, y) in enumerate(train_loader):
        
        # optimization zero grad
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X)

        # compute loss
        y_pred_class = y_pred.round()
        loss = loss_fn(y_pred_class, y)
        
        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
        
    # print epoch and loss at end of every 10th epoch
    if epoch % 10 == 0:
        current_loss = loss.item()
        print(f'epoch: {epoch}, loss: {current_loss}')
        losses.append(current_loss)
    
    
# %% losses
# plot losses
sns.scatterplot(x=range(len(losses)), y=losses)

# %% test the model
# predict on test set
with torch.no_grad():
    y_test_pred = model(X_test).round()

#%% Naive classifier accuracy
# convert y_test tensor [1, 1, 0] to list of strings '[1. 1. 0.]'
y_test_str = [str(i)  for i in y_test.detach().numpy()]
# get most common class count
most_common_cnt = Counter(y_test_str).most_common()[0][1]
# print naive classifier accuracy
print(f"Naive Classifier accuracy: {most_common_cnt/len(y_test_str)*100}%")



# %% Test accuracy
# get test set accuracy
accuracy_score(y_test, y_test_pred)

# %%
