# Title: YOLOv5 vs Yolov8: Unveiling State-of-the-Art Object Detection at 140 FPS.

## Introduction:
Experience the cutting-edge capabilities of YOLOv5, the state-of-the-art object detection model boasting an impressive speed of 140 frames per second. Backed by accurate and reproducible speed benchmarking results, YOLOv5 piqued my interest, prompting an in-depth assessment of its result quality and inference speed. Let's delve into the details!


## The YOLOv5 Model:
   - Implemented in PyTorch, the YOLOv5 model offers ease of understanding, training, and deployment. Architecturally akin to YOLO-v4, a notable distinction lies in the utilization of Cross Stage Partial Network (CSP) to curtail computation costs. Training and running inferences with this model proved remarkably straightforward, and the release includes five model sizes: YOLOv5s (smallest), YOLOv5m, YOLOv5l, and YOLOv5x (largest).
   
   - Setup: Follow **setup.ipynb**
        1. Download the model Yolov5 
            ```
            - git clone https://github.com/ultralytics/yolov5
            ```
        
        2. Install dependencies
            - Install Python version >= 3.10 and pip latest.
            - Create virtual environment.
            ```
            - python -m venv venv
            ```
            - Activate virtual environment.
            ```
            - source venv/bin/activate
            ```
            - Install yolov5 requirenments.
            ```
            - pip install -qr yolov5/requirements.txt
            - pip freeze
            ```
        
        3. With CPU-configuration continue with current dependencies.
        
        4. With GPU-configuration.
            ```
            - pip uninstall torch torchvision torchaudio
            ```
        5. Find your available cuda version on terminal write.
            ```
            - nvidia-smi
            ```
        6. Download CUDA for torch from [here](https://pytorch.org/get-started/previous-versions/) based on your system requirements.
            ```
            - pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
            ```

## FSOCO data:
   - The FSOCO dataset consists of manually annotated images that have been submitted by the Formula Student Driverless community. We provide ground truth data for cone detection and support both bounding boxes and instance segmentation.


## Training with YOLO-v5 on the FOSOCO Dataset:
   - My experience involved training YOLO-v5 on the [FSOCO dataset](https://www.fsoco-dataset.com/download), particularly valuable for autonomous driving applications. This dataset comprises images featuring safety cones strategically placed on roads or footpaths to redirect traffic safely. These cones vary in colors—blue, large orange, orange, unknown, and yellow. The primary goal is to train models on this dataset for subsequent testing. The dataset is divided into training, testing, and validation sets, each with corresponding labeled datasets. The distribution includes 70% training images and labels, 30% testing images and labels, and a validation dataset sourced from the training set.
   
   - Data preparation:
        - Open **data_preprocessing.ipynb**. Divide the data in train, test, validation. Yolo-v5 requires config file in **.yaml** format. 
        - Place the train, test, valid and data.yaml inside models data folder.
        - Hierarchy follows:
            ```
                Repo
                │───data_preprocessing.ipynb 
                │───setup.ipynb 
                │───yolov5-custom-data.ipynb
                │───README.md
                │───yolov5m6.pt
                └───yolov5/
                │   └─data/
                │   │  └─train/
                │   │  │   └─images/ (.jpg)
                │   │  │   └─labels/ (.txt)
                │   │  └─test/
                │   │  │   └─images/ (.jpg)
                │   │  │   └─labels/ (.txt)
                │   │  └─valid/
                │   │  │   └─images/ (.jpg)
                │   │  │   └─labels/ (.txt)
                │   │  └─data.yaml
            ```
   

## Language and Hardware Configuration:
   - Python 3.10 serves as the primary language for model implementation, utilizing CUDA 12.1 with an NVIDIA GeForce RTX 2070 for training. PyTorch with CUDA 12.1 compatibility was employed.
    
## Model Training, Validation and Testing:
Base Model (pre-trained yolov5 checkpoints) is: **yolov5m6.pt**

Open **yolov5-custom-data.ipynb**
- `Training`
    ```
    python yolov5/train.py --batch $TRAIN_BATCH --epochs $TRAIN_EPOCHS --data "data.yaml" --weights $BASE_MODEL --project $PROJECT_NAME --name 'feature_extraction' --cache --freeze 12
    ```
    
- `Validation`
    ```
    WEIGHTS_BEST = "best.pt"  --> weights saved in best.pt
    ```
    ```
	python yolov5/val.py --weights $WEIGHTS_BEST --batch $VAL_BATCH --data 'data.yaml' --task test --project $PROJECT_NAME --name 'validation_on_test_data' --augment
    ```
    
- `Testing`
    ```
    python yolov5/detect.py --weights $WEIGHTS_BEST --conf 0.6 --source 'yolov5/data/test/images' --project $PROJECT_NAME --name 'detect_test' --augment --line=3
    ```
    
- `Inference`
    ```
    WEIGHTS_BEST = f"{PROJECT_NAME}/feature_extraction/weights/best.pt"
    python yolov5/detect.py --source 'yolov5/data/test/images' --weights $WEIGHTS_BEST --conf-thres 0.4
    ```

## Training with YOLO-V8 on the FOSOCO Dataset:

- Data preparation:
    - Open **data_preprocessing.ipynb**. Divide the data in train, test, validation. Yolo-v5 requires config file in **.yaml** format. 
    - Place the train, test, valid and data.yaml inside models data folder.
    - Hierarchy follows:
        ```
            datasets
            │───data.yaml
            │───yolov8m.pt
            │───yolov5m6.pt
            └───data/
            │   │  └─train/
            │   │  │   └─images/ (.jpg)
            │   │  │   └─labels/ (.txt)
            │   │  └─test/
            │   │  │   └─images/ (.jpg)
            │   │  │   └─labels/ (.txt)
            │   │  └─valid/
            │   │  │   └─images/ (.jpg)
            │   │  │   └─labels/ (.txt)
        ```

        ```
        * Before start training move to datasets/ folder.
        ```

    - `Training`
        ```
        yolo task=detect mode=train model=yolov8m.pt data=data.yaml epochs=100 imgsz=640 batch=16 plots=True name=train_complete device=1 project=yolov8 patience=0 lr0=0.00334 lrf=0.15135 momentum=0.74832 weight_decay=0.00025 warmup_epochs=3.3835 warmup_momentum=0.59462 warmup_bias_lr=0.18657 box=0.02 cls=0.21638 dfl=0.15 pose=12.0 kobj=0.51728 label_smoothing=0.0 nbs=64 hsv_h=0.01041 hsv_s=0.54703 hsv_v=0.27739 degrees=0.0 translate=0.04591 scale=0.75544 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.5 mosaic=0.85834 mixup=0.04266 copy_paste=0.0 auto_augment=randaugment erasing=0.04 crop_fraction=1.0 optimizer=Adamax close_mosaic=0
        ```

    - `Validation`
        ```
        yolo detect val model=yolov8/train_complete/weights/best.pt plots=True device=1 data=data.yaml save_json=True save_hybrid=True
        ```
    
## Best Weights for YoloV5 and YolV8:

- `Yolo-V5`
    ```
    train_trainvalid_dist_augment_voc/fine-tuning-background-augment/weights/best.pt
    ```

- `Yolo-V8`
    ```
    datasets/yolov8/train_complete/weights/best.pt
    ```

## .sh Files:
.sh files are added that helps to move the large date around, they can be modified as per requirements.


## Conclusion:
The conclusive analysis reveals YOLOv5's superior run speed, operating approximately 2.5 times faster while excelling in detecting smaller objects. The results exhibit cleaner outputs with minimal overlapping boxes. Kudos to Ultralytics for open-sourcing YOLOv5, a model that seamlessly combines ease of training with efficient inference. This underscores the emerging trend in computer vision object detection, emphasizing models that prioritize both speed and accuracy. If you've experimented with YOLOv5, we welcome you to share your insights in the issues section. Our study focused on training a model using fsoco data, yielding noteworthy results. Notably, we observed an accuracy loss for background instances during training. To address this, we incorporated 10percent of background images into the training dataset and fine-tuned the model over 100 epochs. This strategic augmentation led to enhanced learning, resulting in improved model performance. Further refinement was achieved by introducing augmented crops, specifically targeting instances with limited representation. This step was pivotal in mitigating the risk of overfitting to certain classes arising from an imbalanced dataset. Post-training, a significant improvement in model eﬀicacy was observed. Subsequently, we conducted tests on objects positioned both closer and farther away from the car, employing the minimum distance between the center of the image and the object. This comprehensive evaluation provided insights into the model’s robustness across varying object distances.Parallel experiments were conducted using YOLOv8, acknowledging its higher resource demands in terms of time and memory during training. However, the results obtained justified the investment, showcasing superior performance in both training and validation phases. Key hyperparameters, referred to as golden hyperparameters, were carefully curated and documented to ensure reproducibility. Additionally, the best-weight file resulting from this experiment has been included for reference. This meticulous approach, encompassing data augmentation, fine-tuning, and
strategic testing, along with detailed documentation of hyperparameters and model weights, enhances the reliability and reproducibility of our findings, contributing to the advancement of object detection models.


