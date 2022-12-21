# Training

1. get YOLO project

git clone https://github.com/WongKinYiu/yolov7.git

2. get weights and store in yolov7 folder

wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt

3. adapt cfg file

create copy of yolov7\cfg\training\yolov7-e6e.yaml
set number of classes

4. adapt data file

under data/masks.yaml

5. Get raw images and labels

download from Kaggle
source: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

6. Data Prep 

- Convert Pascal Voc labels to Yolo format
- split data into subfolders

7. perform training

python train.py --weights yolov7-e6e.pt --data data/masks.yaml --workers 1 --batch-size 4 --img 416 --cfg cfg/training/yolov7-masks.yaml --name yolov7 --epochs 5

8. Detection

python detect.py --weights runs/train/yolov7/weights/best.pt 	--conf 0.4 --img-size 640 --source ./test/images/maksssksksss824.png