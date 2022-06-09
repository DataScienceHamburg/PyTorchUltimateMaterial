#%% Packages
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

# %% Data Import
data = sns.load_dataset("flights")
print(f'Number of Entries: {len(data)}')
data.head()

# %%
sns.lineplot(data.index, data.passengers, data=data)
# %%
# Convert passenter data to float32 for PyTorch
num_points = len(data)
Xy = data.passengers.values.astype(np.float32)

#%% scale the data
scaler = MinMaxScaler()

Xy_scaled = scaler.fit_transform(Xy.reshape(-1, 1))


# %% Data Restructuring
X_restruct = [] 
y_restruct = [] 

for i in range(num_points-10):
     list1 = []
     for j in range(i,i+10):
         list1.append(Xy_scaled[j])
     X_restruct.append(list1)
     y_restruct.append(Xy_scaled[j+1])
X_restruct = np.array(X_restruct)
y_restruct = np.array(y_restruct)

#%% train/test split
last_n_months = 12
clip_point = len(X_restruct) - last_n_months
X_train = X_restruct[:clip_point]
X_test = X_restruct[clip_point:]
y_train = y_restruct[:clip_point]
y_test = y_restruct[clip_point:]

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# %%
class FlightDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Dataloader
train_loader = DataLoader(FlightDataset(X_train, y_train), batch_size=2)
test_loader = DataLoader(FlightDataset(X_test, y_test), batch_size=len(y_test))


# %%
class FlightModel(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(FlightModel, self).__init__()
        self.hidden_size = 50
        self.lstm = nn.LSTM(input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=output_size)
    
    def forward(self, x):
        output, _ = self.lstm(x)    
        output = output[:, -1, :]
        output = self.fc1(torch.relu(output))
        return output

# %% Model, Loss and Optimizer
model = FlightModel()

loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
NUM_EPOCHS = 200

#%% Train
for epoch in range(NUM_EPOCHS):
    for j, data in enumerate(train_loader):
        X, y = data
       
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fun(y_pred, y)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.data}")


# %% Create Predictions
test_set = FlightDataset(X_test, y_test)
X_test_torch, y_test_torch = next(iter(test_loader))
with torch.no_grad():
    y_pred = model(X_test_torch)
y_act = y_test_torch.numpy().squeeze()
x_act = range(y_act.shape[0])
sns.lineplot(x=x_act, y=y_act, label = 'Actual',color='black')
sns.lineplot(x=x_act, y=y_pred.squeeze(), label = 'Predicted',color='red')

# %% correlation plot
sns.scatterplot(x=y_act, y=y_pred.squeeze(), label = 'Predicted',color='red', alpha=0.5)

# %%