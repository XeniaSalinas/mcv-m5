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
 
## Documentation
 - Report in Overleaf [link](https://www.overleaf.com/read/dndkxjrdrrzb)
 - Presentation in Google Slides [link](https://docs.google.com/presentation/d/172037oHvwqqKpi6Bd6sYmrkcISgEO9Ft_OU5IjWsbkY/edit?usp=sharing)
 - Reference papers summaries [link](https://github.com/XeniaSalinas/mcv-m5/blob/master/summaries.md)
 - Experiments results [link](https://drive.google.com/drive/folders/0B2fYuDqzasf7TU04SFpfNTh2dzA)
