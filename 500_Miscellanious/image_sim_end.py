#%% packages
from datasets import load_dataset, list_datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from PIL import Image
import os

#%% load model
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

#%% preprocess image function
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# %% load image
image_path = '../070_CNN_ObjectDetection/images/'
image_files = os.listdir(image_path)
img = Image.open(image_path + image_files[0]).convert('RGB')

preprocess(img).unsqueeze(0).shape

#%% create embeddings for candidtate images
embeddings = []
for i in range(100):
    img = Image.open(image_path + image_files[i]).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        embedding = model(img_tensor)
        embedding = embedding[0, :, 0, 0]
    
    embeddings.append(embedding)
#%% compare embeddings to target image
sample_img = Image.open(image_path + image_files[101]).convert('RGB')
img = preprocess(sample_img).unsqueeze(0)
with torch.no_grad():
    sample_embedding = model(img)
    sample_embedding = sample_embedding[0, :, 0, 0]


# %% calculate cosine similarity
similarities = []
for i in range(len(embeddings)):
    # calculate cosine similarity
    similarity = torch.cosine_similarity(sample_embedding, embeddings[i], dim=0).tolist()
    # euclidean distance
    # similarity = torch.dist(sample_embedding, embeddings[i], p=2)
    similarities.append(similarity)

# %%
idx_max_similarity = similarities.index(max(similarities))
image_files[idx_max_similarity]
# %%
