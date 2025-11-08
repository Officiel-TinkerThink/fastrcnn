Faster R-CNN Implementation in Pytorch
========

This repository implements with training, inference and map evaluation in pytorch.
the aim of this simple implementations is to learn and understand the low level concept of the faster r-cnn.

This repo focus on implementation, not on perfomance and metric due to limited hardware resources. It's a fully working pipeline for training and inference, but not yet optimized and having a documented metric performance yet due to that limitation.

The implementation caters to batch size of 1 only and uses roi pooling on single scale feature map.
The repo is meant to train faster r-cnn on voc dataset. Specifically trained on VOC 2007 dataset.


## Data preparation
For setting up the VOC 2007 dataset:
* Download VOC 2007 train/val data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and name it as `VOC2007` folder
* Download VOC 2007 test data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and name it as `VOC2007-test` folder
* Place both the directories inside the root folder of repo according to below structure
    ```
    FasterRCNN-Pytorch
        -> dataset
            -> VOC2007_sampling
            -> VOC2007
                -> JPEGImages
                -> Annotations
            -> VOC2007-test
                -> JPEGImages
                -> Annotations
            -> dataset.py
        -> scripts
            -> train.py
            -> infer.py
        -> config
            -> voc.yaml
            -> voc_trial.yaml
        -> src
            -> model
                -> faster_rcnn.py
                -> roi.py
                -> rpn.py
            -> utils.py
    ```


## Differences from Faster RCNN paper
This repo has some differences from actual Faster RCNN paper.
* Caters to single batch size
* Uses a randomly initialized fc6 fc7 layer of 1024 dim.
* To improve the results one can try the following:
  * Use VGG fc6 and fc7 layers
  * Tune the weight of different losses
  * Experiment with roi batch size
  * Experiment with hard negative mining

# Quickstart
* Create a new conda environment with python 3.8 then run below commands
* ```git clone ```
* ```cd faster_r-cnn```
* ```pip install -r requirements.txt```
* For training/inference use the below commands passing the desired configuration file as the config argument.
* ```python -m scripts.train``` for training Faster R-CNN on voc dataset
* ```python -m scripts.train --trial_mode True``` for training Faster R-CNN on voc dataset with trial mode (using 2 images and one epoch)

* ```python -m scripts.infer --evaluate False --infer_samples True``` for generating inference predictions for one sample image
* ```python -m scripts.infer --evaluate False --infer_samples True --num_samples n``` for generating inference predictions for n images in test dataset
* ```python -m scripts.infer --evaluate True --infer_samples False``` for evaluating on test dataset



## Configuration
* ```config/voc.yaml``` - Allows you to play with different components of faster r-cnn on voc dataset  

* ```config/voc_trial.yaml``` - This would allow you to test the training or inference pipeline with a very small dataset


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created

During training of FasterRCNN the following output will be saved 
* Latest Model checkpoint in ```task_name``` directory

During inference the following output will be saved
* Sample prediction outputs for images in ```task_name/samples/*.png``` 
