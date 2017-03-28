# Scene Understanding for Autonomous Vehicles	

## Team 3: NETSPRESSO
### Team Members
 - Cristina Bustos Rodriguez: [cristinabustos16](https://github.com/cristinabustos16) - <mcb9216@gmail.com>  
 - Lidia Garrucho Moras: [LidiaGarrucho](https://github.com/LidiaGarrucho) - <lidiaxu3@gmail.com>
 - Xian López Álvarez: [xianlopez](https://github.com/xianlopez) - <lopezalvarez.xian@gmail.com>
 - Xènia Salinas Ventalló: [XeniaSalinas](https://github.com/XeniaSalinas) - <salinasxenia@gmail.com>

## Scope
This project aims to design and implement deep neural networks 
for scene understanding for autonomous vehicles.
The first stage of this project is based on object recognition. We aim to classify traffic signals according to its meaning and secene crops depending on the correspondance to a pedestrains, cyclist or vehicles.

## Documentation
 - Report in Overleaf [link](https://www.overleaf.com/read/dndkxjrdrrzb)
 - Presentation in Google Slides [link](https://docs.google.com/presentation/d/172037oHvwqqKpi6Bd6sYmrkcISgEO9Ft_OU5IjWsbkY/edit?usp=sharing)
 - Reference papers summaries [link](https://github.com/XeniaSalinas/mcv-m5/blob/master/summaries.md)
 - Experiments results [link](https://drive.google.com/drive/folders/0B2fYuDqzasf7TU04SFpfNTh2dzA)

# Week2

## Goals

### VGG16
- Train from scratch a VGG16 network using TT100K dataset.
- Data transfer learning on the Belgium Dataset:
 Fine-tune the last FC layers of a VGG16 for the Belgium dataset using the weights obtained training the same VGG16 on TT100K dataset.
 - Train versus fine-tuning performance of the VGG16 network using KITTI dataset.
### ResNet
 - Train from scratch a ResNet network using TT100K dataset.
 - Fine-tuning of the ResNet network using the weights trained on ImageNet.
### InceptionV3
 - 	Train from scratch an InceptionV3 network using TT100K dataset.
 - 	Fine-tuning of the InceptionV3 network using the weights trained on ImageNet.
### DenseNet
 -	Implement the DenseNet network.
 -	Train from scratch a DenseNet using TT100K dataset.
### Boost the performance
 - 	Train the VGG16 on TT100K dataset using data augmentation.
 
## Contributions 
- Adaptation of ResNet network model provided by Keras to the framework in `code/models/resNet.py`
- Adaptation of InceptionV3 network model provided by Keras to the framework in `code/models/inceptionV3.py`
- Implementation from scratch of DenseNet network model in `code/models/denseNet.py`

## Usage

For running this commands, please locate in folder `mcv-m5/code`.

In general, for running the code use `python train.py -c config/config_file_name.py -e experiment_name` where `config_file_name` is your config file and `experiment_name` is the folder name where your experiment is going to be saved. 

If you have a GPU and CUDA installed, before each command put  `CUDA_VISIBLE_DEVICES=0`.


- VGG16

	- TT100K Dataset:
		- Train from scratch: `python train.py -c ./config/classification/tt100k_vgg16_scratch.py -e tt100k_vgg16_scratch`
		- Fine tuning: `python train.py -c ./config/classification/tt100k_vgg16_finetuning.py -e tt100k_vgg16_finetuning`
	
	- Belgium Dataset
		- Transfer Learning: `python train.py -c ./config/classification/belgium_vgg_taska.py -e belgium_vgg_taska`
		
	- KITTI Dataset
		- Train from scratch: `python train.py -c ./config/classification/kitti_vgg16_taskB_scratch.py -e kitti_vgg16_taskB_scratch`
		- Fine tuning: `python train.py -c ./config/classification/kitti_vgg16_taskB_finetuning.py -e kitti_vgg16_taskB_finetuning`
		
- ResNet
	
	- TT100K Dataset:
		- Train from scratch: `python train.py -c ./config/classification/tt100k_resnet_scratch.py -e tt100k_resnet_scratch`
		- Fine tuning: `python train.py -c ./config/classification/tt100k_resnet_finetuning.py -e tt100k_resnet_finetuning`
		
- InceptionV3

	- TT100K Dataset:
		- Train from scratch: `python train.py -c ./config/classification/tt100k_inception_taskd_1.py -e tt100k_inception_taskd_1`
		- Fine tuning: `python train.py -c ./config/classification/tt100k_inception_taskd_finetuning.py -e tt100k_inception_taskd_finetuning`

- DenseNet

	- TT100K Dataset:
		- Train from scratch: `python train.py -c ./config/classification/tt100k_denseNet_taskD_scratch.py -e tt100k_denseNet_taskD_scratch`
		
## Results

The summary of week 2 results can be found in this [link](https://docs.google.com/presentation/d/18LAQ4oPBahXFwfIXP-z77KeSHZ3ZdCD9sSPX-1ivDng/edit?usp=sharing)		

The weights resulting of our experiments can be found in this [link](https://drive.google.com/drive/folders/0B2fYuDqzasf7TU04SFpfNTh2dzA)

# Week 3/4

## Goals

### YOLOv2
- Fine-tune a YOLOv2 network on TT100K dataset.
- Fine-tune a YOLOv2 network on Udacity dataset.
- Evaluate the performance using the F-score and FPS of the best epochs.

### Tiny YOLO
- Fine-tune a Tiny YOLO network on TT100K dataset.
- Fine-tune a Tiny YOLO network on Udacity dataset.
- Evaluate the performance using the F-score and FPS of the best epochs.
 
 ### SSD
- Fine-tune a SSD network on TT100K dataset.
- Fine-tune a SSD network on Udacity dataset.
- Evaluate the performance using the F-score and FPS of the best epochs.

### CAM
- (Not fullfiled) Implement the Class Activation Map technique, that converts any classification network into one that can be used for object localization (without training on bounding boxes).

### Boost the performance
 - 	Train the YOLOv2 on TT100K dataset using data augmentation.
 
## Contributions 
- Added new config files to run the YOLOv2, tiny YOLO and SSD models in `code/config/detection/`.
- Adaptation of SSD model provided by https://github.com/rykov8/ssd_keras to the framework in `code/models/ssd.py` and `code/layers/ssd_layers.py`.
- Adaptation of SDD Loss and metrics provided by https://github.com/rykov8/ssd_keras to the framework in `code/metrics/metrics.py`.
- Adapt F-score and FPS evaluation code to SSD model in `code/eval_detection_fscore.py`.
- Added SSD utils in `code/tools/sdd_utils.py` and `code/tools/detection_utils.py`.

## Usage

For running this commands, please locate in folder `mcv-m5/code`.

In general, for running the code use `python train.py -c config/config_file_name.py -e experiment_name` where `config_file_name` is your config file and `experiment_name` is the folder name where your experiment is going to be saved. 

If you have a GPU and CUDA installed, before each command put  `CUDA_VISIBLE_DEVICES=0`.

- YOLOv2

	- TT100K_detection Dataset:
		- Fine tuning: `python train.py -c ./config/detection/tt100k_detection.py -e tt100k_yolo_taskA`
	
	- Udacity Dataset
		- Fine tuning: `python train.py -c ./config/detection/udacity_yolo_taskC.py -e udacity_yolo_taskC`
		
	- Data augmentation
		- Fine tuning: `python train.py -c ./config/detection/tt100k_detection_dataAug_taskE.py -e tt100k_detection_dataAug_taskE`
		
- Tiny YOLO
	- TT100K_detection Dataset:
		- Fine tuning: `python train.py -c ./config/detection/tt100k_tiny_yolo_taskA.py -e tt100k_tiny_yolo_taskA`
	
	- Udacity Dataset
		- Fine tuning: `python train.py -c ./config/detection/udacity_tiny_yolo_taskC.py -e udacity_tiny_yolo_taskC`

- SSD
	- TT100K_detection Dataset:
		- Fine tuning: `python train.py -c ./config/detection/tt100k_detection_ssd.py -e tt100k_ssd_taskD`
	
	- Udacity Dataset
		- Fine tuning: `python train.py -c ./config/detection/udacity_ssd_taskD.py -e udacity_ssd_taskD`
		
- CAM
	- This technique is not totally operative yet. Its code is in the following files:
		- predict_cam.py: this is the executable file, that loads an image and calls the rest of the functions related.
		- tools/cam_utils.py: here are the functions that use the network to predict, compute the heatmap, and build the bounding boxes.
		- models/vggGAP.py: modification of VGG-16 network that is used during the training phase. We adapted the dataset TT100K_detection for classification (also incrementing the validation set), and trained there this network.
		- models/vggGAP_pred.py: this is the version of the network that is used during the prediction phase. It loads the weights we obtained during trainig, and produces a heatmap.

## Results

The summary of week 3/4 results can be found in this [link](https://docs.google.com/presentation/d/18LAQ4oPBahXFwfIXP-z77KeSHZ3ZdCD9sSPX-1ivDng/edit?usp=sharing).	

The weights resulting of our experiments can be found in this [link](https://drive.google.com/open?id=0B2fYuDqzasf7Y3dWSTgwT25wZkk).