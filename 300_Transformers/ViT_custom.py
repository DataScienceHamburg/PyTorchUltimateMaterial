#%%
# source: https://medium.com/@yanis.labrak/how-to-train-a-custom-vision-transformer-vit-image-classifier-to-help-endoscopists-in-under-5-min-2e7e4110a353
import pandas
#%%
#%% Packages
from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import ViTFeatureExtractor, ViTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#%% data prep
train, val, id2label, label2id = VisionDataset.fromImageFolder(
    "./train/",
    test_ratio   = 0.1,
    balanced     = True,
    augmentation = True, 
    torch_vision = False
)

#%%
huggingface_model = 'google/vit-base-patch16-224-in21k'

trainer = VisionClassifierTrainer(
	model_name   = "MyDogClassifier",
	train        = train,
	test         = val,
	output_dir   = "./out/",
	max_epochs   = 20,
	batch_size   = 4, 
	lr	     = 2e-5,
	fp16	     = True,
	model = ViTForImageClassification.from_pretrained(
	    huggingface_model,
	    num_labels = len(label2id),
	    label2id   = label2id,
	    id2label   = id2label
	),
	feature_extractor = ViTFeatureExtractor.from_pretrained(
		huggingface_model,
	),
)

#%% Model Evaluation
y_true, y_pred = trainer.evaluate_f1_score()

#%%
cm = confusion_matrix(y_true, y_pred)
labels = list(label2id.keys())
df_cm = pd.DataFrame(cm, index = labels, columns = labels)

# plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
plt.savefig("./conf_matrix_1.jpg")

# %% Inference
import os.path
path = "./out/MYDOGCLASSIFIER/20_2022-09-09-22-30-04/model/"
img  = "./test/affenpinscher/affenpinscher_0.jpg"

classifier = VisionClassifierInference(
    feature_extractor = ViTFeatureExtractor.from_pretrained(path),
    model = ViTForImageClassification.from_pretrained(path),
)

label = classifier.predict(img_path=img)
print("Predicted class:", label)

# %% Test dataset
test, _, id2label, label2id = VisionDataset.fromImageFolder(
    "./test/",
    test_ratio   = 0,
    balanced     = True,
    augmentation = True, 
    torch_vision = False
)

classifier.predict(img)
# %% 
import glob
test_files = [f for f in glob.glob("./test/**/**", recursive=True) if os.path.isfile(f)]
# %%
for i in range(len(test_files)):
	print(f"{test_files[i]}")
	print(f"predicted: {classifier.predict(test_files[i])}")
# %%
