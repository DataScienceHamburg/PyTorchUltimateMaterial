#%% 
from ultralytics import YOLO

# sources: 
# https://docs.ultralytics.com/cli/
# https://docs.ultralytics.com/cfg/
# %% load the model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# %% Train the model
results = model.train(data="train_custom/masks.yaml", epochs=1, imgsz=512, batch=4, verbose=True, device='cpu')
# device=0...GPU
# %% Export the model
model.export()
# %% 
import torch
# %%
torch.cuda.is_available()
# %%
