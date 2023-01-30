
# Package Installation

- run pip install ultralytics before

```
import ultralytics
# check installation
ultralytics.checks()
```

# YOLO v8 source

(YOLO v8)[https://github.com/ultralytics/ultralytics]

# Train Custom Dataset

- (turn off firewall, optional)

- copy all data folders "train", "val", and "test" to the new subfolder "train_custom"

- download yolov8n.pt to the subfolder

- masks.yml: adapt the paths to absolute paths

On command line run

```
!yolo task=detect mode=train model=yolov8n.pt data=masks.yaml epochs=3 imgsz=150
```

# Inference

On command line run

```
yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source='test/kiki.jpg'
```  

# References

[Ultralytics](https://github.com/ultralytics/ultralytics)