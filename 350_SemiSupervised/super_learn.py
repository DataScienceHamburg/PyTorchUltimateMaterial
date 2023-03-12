#%% packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
from sklearn.metrics import accuracy_score
from PIL import Image
import seaborn as sns
# %% Hyperparameters
BATCH_SIZE = 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 50
# %% image transformation steps
transform_super = transforms.Compose(
    [transforms.Resize(32),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])

#%% Dataset for train and test
train_ds = torchvision.datasets.ImageFolder(root='data/train', transform=transform_super)
test_ds = torchvision.datasets.ImageFolder(root='data/test', transform=transform_super)
# %% Dataloaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

#%% Model Class
class SupervisedNet(nn.Module):
    def __init__(self, n_super_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out_super = nn.Linear(64, n_super_classes)
        self.relu = nn.ReLU()
        self.output_layer_super = nn.Sigmoid()
            
    def backbone(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc_out_super(x)
        x = self.output_layer_super(x)
        return x

model = SupervisedNet(n_super_classes=2)
model.train()
#%% Loss functions and Optimizer
criterion_supervised = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %% Training loop
train_losses = []
for epoch in range(NUM_EPOCHS):
    train_loss = 0
    for i, (supervised_data) in enumerate(train_loader):
        X_super, y_super = supervised_data
        
        # init gradients
        optimizer.zero_grad()
        
        # forward pass
        y_super_pred = model(X_super)
        
        # calc losses
        loss = criterion_supervised(y_super_pred, y_super)
        
        # calculate gradients
        loss.backward()
        
        # update weights
        optimizer.step()
        
        # extract training losses
        train_loss += loss.item()
    train_losses.append(train_loss)
    print(f"Epoch {epoch}: Loss {train_loss}")
# %%
sns.lineplot(x=list(range(len(train_losses))), y=train_losses)
# %%
y_test_preds = []
y_test_trues = []
with torch.no_grad():
    for (X_test, y_test) in test_loader:
         y_test_pred = model(X_test) 
         y_test_pred_argmax = torch.argmax(y_test_pred, axis = 1)
         y_test_preds.extend(y_test_pred_argmax.numpy())
         y_test_trues.extend(y_test.numpy())
# %%
accuracy_score(y_pred=y_test_preds, y_true=y_test_trues)
# %%
