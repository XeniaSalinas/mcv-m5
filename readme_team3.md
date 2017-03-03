# Scene Understanding for Autonomous Vehicles	

## Team 3: NETSPRESSO
### Team Members
 - Cristina Bustos <mcb9216@gmail.com>
 - Lidia Garrucho Moras <lidiaxu3@gmail.com>
 - Xian López Álvarez <lopezalvarez.xian@gmail.com>
 - Xènia Salinas <salinasxenia@gmail.com>

# Scope
This project aims to design and implement deep neural networks 
for scene understanding for autonomous vehicles.

## Documentation
 - Overleaf [link](https://www.overleaf.com/read/dndkxjrdrrzb)


# References


## [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
### Authors: Karen Simonyan & Andrew Zisserman

### Summary


## [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
### Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

### Summary
This paper presents the a very deep network of 152 layers with less computational complexity than a VGG network. At the time it was presented, it was the deepest network tested on ImageNet.
The methodology followed was to add residual learning in the network to ease the optimization during the training and avoid degradation due to the depth. Residual learning is adopted every few stacked layers and implemented as shortcut connections and element-wise additions. The shortcut connection does not introduce extra parameters or computation complexity to the network.  
The residual network is based on a modification of the VGG network of 34 layers with fewer filters and lower complexity. The identity shortcuts are added every two layers after the first pooling. 

The ImageNet implementation uses scale and color aumentation on images or its horizontal flip randomly cropped to 224x224. Batch normalization is applied after each convolution and before activation. The network is trained from scratch using a SGD batch size of 256 and a start learning rate of 0.1, divided by 10 when riching a plateau.
The model is training during 600.000 epochs using a weight decay of 0.0001 an a momentum of 0.9.

Their experiments showed that the deeper plain network of 34 layers, inspired on VGG, has greater validation error than a reduce  version of 18 layers. They conjecture that the reason may be that the convergence rate in deep plain nets decrease exponentially.
The next experiment was to validate 18-layer and 34-layer residual nets with identity mapping shortcuts to compare the results with their plain versions. The 34-layer model has lower training error than the 18-layer and improves the results of their plain versions, having addressed the degradation problem and gaining in accuracy. It was also showed that a 18-layer residual network converges faster than the plain version. We can conclude that ResNet eases optimization by providing faster convergence at the early stage.
The effect of the projection shortcuts is analized using three different configurations. It was shown that the configuration that uses some projection shortcuts to increase dimensions and identity shortcuts for not to increase the complexity is the best one.

The net presented on ImageNet 2012 was a deep bottleneck architecture of the previous 34-layer residual net. The identity shortcuts are added every 3 layer instead of 2. The bottleneck model is a stack of 1x1, 3x3 and 1x1 convolutions. In a 50-layer ResNet, the 2-layer blocks are replaced with the bottleneck configuration adding a total of 50 layers. The 101-layer and 152-layer models are constructed using extra layers in the 3-layer blocks. The results showed that the 50/101/152-layer networks are more accurate than the 34-layer residual networks and have lowe complexity than VGG-16/19 ntes.
