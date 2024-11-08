## GAN Exercise With 03 Color Channels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

##################### Image Creation #####################
import math
from random import uniform
import seaborn as sns

## Create the plot and save as image
#sns.set(rc={'figure.figsize':(12,12)})
#sns.set(rc={'figure.figsize':(5,5)})

#TRAIN_DATA_COUNT = 1024
#theta = np.array([uniform(0, 2 * np.pi) for _ in range(TRAIN_DATA_COUNT)])# np.linspace(0, 2 * np.pi, 100)

##Generating x and y data
#x = 16 * (np.sin(theta) ** 3)
#y = 13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)
#ax = sns.scatterplot(x=x, y=y)
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)
#plt.tight_layout()

#plt.savefig("../data/heart/heart_01.jpg", bbox_inches='tight')

##################### Model Development, Training and Evaluation #####################

##################### Class Discriminator #####################

## in_features -> (Batch_size, Color_channels, Height, Width)
##          or -> (Color_channels, Height, Width)
IN_FEATURES = 3*28*28
OUT_FEATURES = 1

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features=IN_FEATURES, out_features=128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=128, out_features=OUT_FEATURES),## Fake=0, Real=1
            nn.Sigmoid(),
        )
    def forward(self, x):
        return(self.disc(x))

##################### Class Generator #####################

##z_dim is the dimension of the latent noise that the generator is going to take as input
## z_dim Can try 128, 256 and smaller as well
z_dim = 64
img_dim = 3*28*28

class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=256, out_features=img_dim)
        )
    def forward(self, x):
        return self.gen(x)


##################### Transforms, Image Path, Dataset (ImageLoader), DataLoader #####################
#from torch.utils.data import Dataset, DataLoader
transforms = transforms.Compose([
    transforms.Resize((28, 28)),
    #transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

BATCH_SIZE = 1##32

path_images = './data/'
#path_images = '/Users/luis/Documents/Programming/Courses_Programming/0758 PyTorch Ultimate 2023 From Basics to Cutting Edge/venv_0758_PyTorch_From_Basics_Cutting_Edge_310_001/github_DataScienceHamburg/PyTorchUltimateMaterial/220_GAN/data/'

train_dataset = ImageFolder(root=path_images, transform=transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


##################### Hyperparameters #####################

LR = 1e-4
NUM_EPOCHS = 400

##################### Instances #####################
## Loss Function
## Note: Should it still be BCELoss even if it has 3 color channels? ## It is still classifying as right or wrong so I believe it should
criterion = nn.BCELoss()

## Creating instances of the model
image_dim = 3*28*28
z_dim = 64## Latent Space Noise

disc = Discriminator(img_dim=image_dim)
gen = Generator(z_dim=z_dim)

## Optimizers for Discriminator and Generator
optimizer_discriminator = torch.optim.Adam(params=disc.parameters(), lr=LR)
optimizer_generator = torch.optim.Adam(params=gen.parameters(), lr=LR)

##################### Training  #####################
## Training the discriminator and Generator in an alternative fashion

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real_samples, _) in enumerate(train_loader):
        real_samples = real_samples.view(-1, image_dim)

        real_samples_labels = torch.ones((BATCH_SIZE, 1))

        ## Creating random samples
        latent_space_samples = torch.randn((BATCH_SIZE, z_dim))

        generated_samples = gen(latent_space_samples)

        generated_samples_labels = torch.zeros((BATCH_SIZE, 1))
        
        all_samples = torch.cat((real_samples, generated_samples), dim=0)

        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels), dim=0)

        ## Training Discriminator 
        if epoch % 2 == 0:
            disc.zero_grad()
            ## disc Input needs to be img_dim = 784 For 1x28x28
            ## disc Input needs to be img_dim = 2352 For 3x28x28
            output_discriminator = disc(all_samples)
            loss_discriminator = criterion(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()
        
        ## Training Generator
        if epoch % 2 == 1:

            latent_space_samples = torch.randn((BATCH_SIZE, z_dim))
            gen.zero_grad()
            generated_samples = gen(latent_space_samples)
            
            output_discriminator_generated = disc(generated_samples)

            loss_generator = criterion(output_discriminator_generated, real_samples_labels)

            loss_generator.backward()
            optimizer_generator.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch: [{epoch}/{NUM_EPOCHS}]")
            ## Note: Sometimes it works when I comment out the next line
            #print(f"Epoch: [{epoch}/{NUM_EPOCHS}], Discriminator Loss: {loss_discriminator}, Generator Loss: {loss_generator}")## Commenting this line makes it work
            print(f"loss_discriminator: {loss_discriminator:.6f}")
            #print(f"loss_generator: {loss_generator:.6f}")
            #print("")
            with torch.no_grad():
                latent_space_samples = torch.randn((BATCH_SIZE, z_dim))
                generated_samples = gen(latent_space_samples)
                generated_samples_processed = generated_samples.view(3, 28, 28).detach()
                generated_samples_processed = 0.5 * (generated_samples_processed +1)## Denormalize
                generated_samples_processed = generated_samples_processed.view(28, 28, 3)
                plt.imshow(generated_samples_processed, cmap='viridis')##cmap='gray', 'bone'
                plt.grid(False)
                plt.savefig(f"./saved_images/saved_fig_GAN_03Ch_{epoch}_{NUM_EPOCHS}.png")