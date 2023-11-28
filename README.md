# Title: YOLOv5: Unveiling State-of-the-Art Object Detection at 140 FPS.

## Introduction:
Experience the cutting-edge capabilities of YOLOv5, the state-of-the-art object detection model boasting an impressive speed of 140 frames per second. Backed by accurate and reproducible speed benchmarking results, YOLOv5 piqued my interest, prompting an in-depth assessment of its result quality and inference speed. Let's delve into the details!


## The YOLOv5 Model:
   - Implemented in PyTorch, the YOLOv5 model offers ease of understanding, training, and deployment. Architecturally akin to YOLO-v4, a notable distinction lies in the utilization of Cross Stage Partial Network (CSP) to curtail computation costs. Training and running inferences with this model proved remarkably straightforward, and the release includes five model sizes: YOLOv5s (smallest), YOLOv5m, YOLOv5l, and YOLOv5x (largest).

## FSOCO data:
   - The FSOCO dataset consists of manually annotated images that have been submitted by the Formula Student Driverless community. We provide ground truth data for cone detection and support both bounding boxes and instance segmentation.


## Training with YOLO-v5 on the FOSOCO Dataset:
   - My experience involved training YOLO-v5 on the [FSOCO dataset](https://www.fsoco-dataset.com/download), particularly valuable for autonomous driving applications. This dataset comprises images featuring safety cones strategically placed on roads or footpaths to redirect traffic safely. These cones vary in colors—blue, large orange, orange, unknown, and yellow. The primary goal is to train models on this dataset for subsequent testing. The dataset is divided into training, testing, and validation sets, each with corresponding labeled datasets. The distribution includes 70% training images and labels, 30% testing images and labels, and a validation dataset sourced from the training set.

## Language and Hardware Configuration:
   - Python 3.10 serves as the primary language for model implementation, utilizing CUDA 12.1 with an NVIDIA GeForce RTX 2070 for training. PyTorch with CUDA 12.1 compatibility was employed.
    
## Model Training, Validation and Testing:
- `Training`
    ```
    python3.10 yolov5/train.py --batch $TRAIN_BATCH --epochs $TRAIN_EPOCHS --data "data.yaml" --weights $BASE_MODEL --project $PROJECT_NAME --name 'feature_extraction' --cache --freeze 12
    ```
    
- `Validation`
    ```
    WEIGHTS_BEST = "best.pt"  --> weights saved in best.pt
    ```
    ```
	python3.10 yolov5/val.py --weights $WEIGHTS_BEST --batch $VAL_BATCH --data 'data.yaml' --task test --project $PROJECT_NAME --name 'validation_on_test_data' --augment
    ```
    
- `Testing`
    ```
    python3.10 yolov5/detect.py --weights $WEIGHTS_BEST --conf 0.6 --source 'yolov5/data/test/images' --project $PROJECT_NAME --name 'detect_test' --augment --line=3
    ```
    
## Conclusion:
The conclusive analysis reveals YOLOv5's superior run speed, operating approximately 2.5 times faster while excelling in detecting smaller objects. The results exhibit cleaner outputs with minimal overlapping boxes. Kudos to Ultralytics for open-sourcing YOLOv5, a model that seamlessly combines ease of training with efficient inference. This underscores the emerging trend in computer vision object detection, emphasizing models that prioritize both speed and accuracy. If you've experimented with YOLOv5, we welcome you to share your insights in the issues section.



