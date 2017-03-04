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
In this paper, the effect of the convolutional network depth on accuracy is studied. The main contribution is the evaluation of the network depth, by adding more convolutional layers using small convolutional filters of 3x3, achieving better results when depth is 16-19 weight layers. As a result, the accuracy was significantly higher respect the state-of-the-art accuracy on ILSVRC tasks, furthermore, the architecture was also applicable to other image recognition tasks. 

To the convNet, the input is a 224x224 image, with the RGB mean subtracted in every pixel. For each convolutional layer, a 3x3 receptive field is used, the convolution stride is 1 and the padding is 1. Some conv. layers are followed by a 2x2 max pooling with stride 2.
Several convolutional layer depths were tested, to 11 weight layer to 19. In each depth tested, the conv. layer width started from 64 in the first layer and increasing by a 2 factor after max pooling until it reaches 512.  For each of the depth tested, the convolutional layers are followed by 3 fully connected layers, the first 2 have 4096 elements and the last one has 1000.  The final layer is a soft-max layer. All hidden layers have the rectification non-linearity ReLU.
The improvements of this architecture respect to the top-performing entries of ILSVRC are: firstly, for 3 conv. layers, they incorporate 3 non-linear rectification layers instead of 1, making the decision function more discriminative, second, the incorporation of 1x1 conv. layers increases the decision function non-linearity.

The training consists in optimize the multinomial logistic regression objective. The batch size was 256. The back propagation used a momentum of 0.9. The L2 penalty multiplier for regularization by weight decay was 5x10^-4. Dropout of 0.5 for the first two fc layers. The initial learning rate was 10^-2, and decreases by a 10 factor when the validation set accuracy stops improving. The learning stopped after 370K iterations (74 epochs).
For the weight initialization, they started training the configuration A (only 11 conv. layers) with random weights. For deeper architecture, they initialize the first four conv. layers and last three fc layers with the resulting weights of training the configuration A, the other layers were initialized randomly. Those weights can change during the training process. The biases were initialized with zero.
For the training set, the input of the convNet is a 224x224 image. If the scale size of the training image is 224, the whole image is taken. If the scale size of the image is greater than 224, two approaches are considered: single-scale and multi-scale. The first one, single-scale, the scale size is fixed to 384 or 256. In multi-scale, the scale size is randomly chosen between a range of 256 to 512, this permits that a single model is trained to recognize objects over a wide range of scales.

At testing time, the test image is scaled using scale size not necessarily equal to the scale using in training. Then the network is applied over the test image, the fc layers are converted to conv. layers, and the resulting fc net is applied over the image. The result is a class score map with the number of channels equal to the number of classes. To obtain the score of each class, each score map is spatially averaged.

Experiments were evaluated using top-1 val. error and top-5 val. error. The better classifications results are using multi-crop and dense evaluation, as multi-crop results in a finer sampling of the input image compared to the fully convolutional net, and dense means the padding is done using the neighbor pixels instead of zeros. But in practice, the increased computation time using multi-crops does not provide potential gains in accuracy compared with multi-scale evaluation (test image over several rescale versions).  Posterior experiments showed that combining the output of several models by averaging their soft-max class posteriors increased slightly the accuracy.



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
