#%% packages
from detecto import core, utils
from detecto.visualize import show_labeled_image
from torchvision import transforms
import numpy as np


#%% data download
path_images = 'images'
path_train_labels = 'train_labels'
path_test_labels = 'test_labels'

# %% data augmentation
custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((50)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    utils.normalize_transform()
])

# %% dataset and dataloader
trained_labels = ['apple', 'banana']

train_dataset = core.Dataset(image_folder=path_images, label_data=path_train_labels, transform=custom_transforms)
test_dataset = core.Dataset(image_folder=path_images, label_data=path_test_labels, transform=custom_transforms)

train_loader = core.DataLoader(train_dataset, batch_size=2, shuffle=False)
test_loader = core.DataLoader(test_dataset, batch_size=2, shuffle=False)
# %% initialize model
model = core.Model(trained_labels)
# %% perform the training
losses = model.fit(train_loader, test_dataset, epochs=2, verbose=True)

# %% show image with predictions
test_image_path = 'images/apple_77.jpg'
test_image = utils.read_image(test_image_path)
pred = model.predict(test_image)
labels, boxes, scores = pred
show_labeled_image(test_image, boxes, labels)

#%% show image with predictions above confidence threshold
conf_threshold = 0.7
filtered_indices = np.where(scores > conf_threshold)
filteres_scores = scores[filtered_indices]
filtered_boxes = boxes[filtered_indices]
num_list = filtered_indices[0].tolist()
filtered_labels = [labels[i] for i in num_list]
show_labeled_image(test_image, filtered_boxes, filtered_labels)

# %% get predictions
y_test_pred = []
import torch
with torch.no_grad():
    for j, data in enumerate(test_loader):
        image, label = data
        output = model.predict(image)
        y_test_pred.extend(output)
        
        break

#%% calculate IoU
for j, data in enumerate(train_loader):
    print(j)
    image, label = data
    print(label)
