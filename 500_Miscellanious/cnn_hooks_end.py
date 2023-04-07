#%% packages
import torch
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

#%% Data Prep
image_path = 'kiki.jpg'
image = Image.open(image_path)
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

X = transformations(image).unsqueeze(0)
X.shape
# %%
model = resnet18(pretrained = True)

#%% Hook class
class MyHook:
    def __init__(self) -> None:
        # save layer output
        self.layer_out = []
        # save layer shape
        self.layer_shape = []
        
    def __call__(self, module, module_in, module_out):
        self.layer_out.append(module_out)
        self.layer_shape.append(module_out.shape)
        
#%% Register Hook
my_hook = MyHook()
for l in model.modules():
    if isinstance(l, torch.nn.modules.conv.Conv2d):
        handle = l.register_forward_hook(my_hook)
        

# %% Forward Pass
y_pred = model(X)

# %% check that outputs were created
len(my_hook.layer_out)

#%%
layer_num = 2
layer_imgs = my_hook.layer_out[layer_num].detach().numpy()

for i in range(4):
    plt.imshow(layer_imgs[0, i, :, :])
    plt.show()
# %%
