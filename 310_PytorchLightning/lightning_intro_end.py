#%% packages
import graphlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns


import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

#%% Dataset and Dataloader
class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(dataset = LinearRegressionDataset(X_np, y_np), batch_size=2)



#%%

class LitLinearRegression(pl.LightningModule):
    def __init__(self, input_size, output_size):
        super(LitLinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.loss_fun = nn.MSELoss()
    
    def forward(self, x):
        return self.linear(x)

    def configure_optimizers(self):
        learning_rate = 0.02
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        X, y = train_batch

        # forward pass
        y_pred = self.forward(X)

        # compute loss
        loss = self.loss_fun(y_pred, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def val_step(self, val_batch, batch_idx):
        X, y = val_batch

        # forward pass
        y_pred = model(X)

        # compute loss
        loss = self.loss_fun(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

#%% model instance and training
# model instance
model = LitLinearRegression(input_size=1, output_size=1)

# training
early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=2, verbose=True, mode="min")

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=500, log_every_n_steps=2, callbacks=[early_stop_callback])
trainer.fit(model=model, train_dataloaders=train_loader)

# %% after how many steps the model converged?
trainer.current_epoch

# %% get model parameters
for parameter in model.parameters():
    print(parameter)
# %%
