# Scene Understanding for Autonomous Vehicles	

## Team 3: NETSPRESSO
### Team Members
 - Cristina Bustos Rodriguez: [cristinabustos16](https://github.com/cristinabustos16) - <mcb9216@gmail.com>  
 - Lidia Garrucho Moras: [LidiaGarrucho](https://github.com/LidiaGarrucho) - <lidiaxu3@gmail.com>
 - Xian López Álvarez: [xianlopez](https://github.com/xianlopez) - <lopezalvarez.xian@gmail.com>
 - Xènia Salinas Ventalló: [XeniaSalinas](https://github.com/XeniaSalinas) - <salinasxenia@gmail.com>

# Scope
This project aims to design and implement deep neural networks 
for scene understanding for autonomous vehicles.
The first stage of this project is based on object recognition. We aim to classify traffic signals according to its meaning and secene crops depending on the correspondance to a pedestrains, cyclist or vehicles.

## Goals of Week2
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
		- Train from scratch: `python train.py -c ./config/classification/***.py -e ***`
		- Fine tuning: `python train.py -c ./config/classification/***.py -e ***`
		
- InceptionV3

	- TT100K Dataset:
		- Train from scratch: `python train.py -c ./config/classification/tt100k_inception_taskd_1.py -e tt100k_inception_taskd_1`
		- Fine tuning: `python train.py -c ./config/classification/tt100k_inception_taskd_finetuning.py -e tt100k_inception_taskd_finetuning`

- DenseNet

	- TT100K Dataset:
		- Train from scratch: `python train.py -c ./config/classification/tt100k_denseNet_taskD_scratch.py -e tt100k_denseNet_taskD_scratch`
		
## Documentation
 - Report in Overleaf [link](https://www.overleaf.com/read/dndkxjrdrrzb)
 - Presentation in Google Slides [link](https://docs.google.com/presentation/d/172037oHvwqqKpi6Bd6sYmrkcISgEO9Ft_OU5IjWsbkY/edit?usp=sharing)
 - Reference papers summaries [link](https://github.com/XeniaSalinas/mcv-m5/blob/master/summaries.md)
 - Experiments results [link](https://drive.google.com/drive/folders/0B2fYuDqzasf7TU04SFpfNTh2dzA)
