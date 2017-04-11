# Classification

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


# Detection

## [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
### Authors: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi

### Summary
This paper presents a new approach to object detection where a single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation.

All the previous systems take a classifier for an object and evaluate it at different locations and scales in the image. These pipelines are slow and hard to optimize because each component must be trained separately. YOLO formulates detection as a regression problem, for that it uses a single convolutional network that predicts the bounding box and the class at the same time, this model has some benefits over the previous systems. The first benefit is that is extremely faster, the network runs at 45 fps on a Titan X GPU and at 150 fps with the Fast YOLO while Faster R-CNN is 2.5 times slower. The second benefit is that YOLO has a global image perception making a reduction on the background errors, specifically it is reduced to less than a half compared to Fast R-CNN. The third benefit is that YOLO learns generalizable representations of objects, so the probability of breaking down on new domains is reduced. Another benefit is that this system enables end-to-end training and real time speed. Despite this benefits it also has some limitations. The number of nearby objects that it can predict is limited cause of the strong spatial constraints that YOLO imposes. Another limitation is that it uses coarse features to predict bounding boxes since it contains multiple down-sampling layers. Finally, the loss function equally treats errors is small and large bounding boxes.

The system divides the image into a SxS grid and assigns the responsibility of detecting an object to the grid cell in which falls the center of that object. Thus, each grid cell predicts B bounding boxes (center coordinates respect to the grid cell bounds and width and height relative to the whole image) and the confidence scores (Pr(Object) * IOU between the predicted box and the ground truth) for each one. Moreover, each grid cell predicts C conditional class probabilities, Pr(Class_i|Object). Multiplying the conditional class probability and the individual box confidence prediction they obtain the class-specific confidence score for each box.

The YOLO arquitecture is composed by 24 convolutional layers followed by 2 fully connected layers. It uses 1x1 convolutional layers to reduce the number of features followed by 3x3 convolutional layers. For the Fast YOLO version, they reduce to 9 the number of convolutional layers an also the number of filter in those layers. The output is a 7x7x30 predictions tensor.

They first pretrained the convolutional layers on ImageNet using the first 20 convolutional layers followed by an average-pooling and a fully connected layer. Then they fine-tunned the whole network optimizing the sum-squared error in the output, to deal with the grid cells that do not contain any object they introduced to parameters that increase the loss from bounding box coordinate predictions and decrease the loss from confidence predictions. They trained the network for 135 epochs using a batch size of 64, a momentum of 0.9 and a decay of o.0005. The learning rate slowly raise from 10^-3 to 10^-2, the it is maintained during 75 epochs, after that it is reduced to 10^-3 for 30 epochs and finally to 10^-4 for 30 epochs. They also used dorpout with a 0.5 rate and an expensive data augmentation based on scales, translation and exposure and saturation adjustments.

On the inference step it predicts 98 bounding boxes per image in a single network evaluation. Often is predicts only one box for each object, but for the cases in which it predicts multiple boxes they apply the non-maximal suppression algorithm.

Comparing the results with other real-time systems they saw that Fast YOLO was the fastest object detection method and that with a 52.7% mAP they doubled the accuracy of prior real-time systems. YOLO obtained 63.4% mAP and had real-time performance.

Comparing the results with Fast R-CNN they saw that the most common error in YOLO was localization error while on Fast R-CNN they had three time the probability to predict background detections. They also developed a combination of Fast R-CNN and YOLO where YOLO system eliminates the background detections from Fast R-CNN. With this combination, they achieved a 3.2 gain over the Fast R-CNN getting a 75.0% mAP on the VOC 2007 test set.

On VOC 2012 test set YOLO scores 57.9% mAP, lower than the other state of the art systems. The combined system (Fast R-CNN + YOLO) is one of the highest performing detection. Finally, they analyzed the generalizability of the system testing person detection on artwork, they could see that its AP degrades less than other methods, from 59.2% on VOC 2007 to 53.3% on Picasso dataset or 45% on People-Art.

## [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
### Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg

### Summary 
This paper presents an object detection deep neural network that outperforms state-of-the-art Faster R-CNN model and is faster than other single stage methods such as YOLO. The improvements in speed and accuracy come from eliminating bounding box proposals and the subsequent pixel or feature resampling stage and adding default boxes of different aspect ratios on each feature location for multiple feature maps at different scales. 

The convolutional feature layers to perform multi-scale feature maps are added to the base network (modified VGG-16 and truncated before any classification layers).  Each feature layer can produce a fixed set of detection predictions using a set of convolutional filters. A small convolutional filter is used to predict object categories and offsets in bounding box locations, using different filters for different aspect ratio detections and applying them to multiple feature maps at the last stages of the network to implement detection at multiple scales. 

The base model is an à trous version of VGG-16, 20% faster than the original. The fc6 and fc7 are converted to convolutional layers, the parameters from fc6 and fc7 are subsampled and the pool5 is modified to 3x3 – s1. Additionally, they use the à trous algorithm to fill the holes and they remove the dropout layers and the fc8 layer.

![model ssd](http://joshua881228.webfactional.com/media/uploads/ReadingNote/arXiv_SSD/SSD.png "SSD Model")

SSD is simple to train, it only needs an input image and ground truth boxes for each object during training. Then a small set of default bounding boxes are evaluated with different aspect ratios at each location in several feature maps with different scales. For each box, it predicts the shape offsets and the confidence for all object categories. At training the default boxes are matched with the ground truth boxes and the loss function is composite by the weighted sum between the localization loss (e.g. Smooth L1) and the confidence loss (e.g. Softmax).

In order to determine which default boxes correspond to ground truth detection boxes, the default boxes with the best jaccard overlap are used, as in Multibox. The main difference is that SSD allows the network to predict high scores for multiple overlapping default boxes instead of keeping the one with maximum overlap. That technique also helps to reduce the ratio between the negative and positive training examples which leads to faster optimization and more stable training.
The SSD training objective is also derived from MultiBox but adapted to handle multiple object categories. The localization loss is Smooth L1 loss and the confidence loss is the Softmax loss over multiple classes confidence.

SSD implements data augmentation to make the model more robust and improving up to 8.8% mAP the testing results. Each training image is randomly sampled by: using the original input image, sampling a patch so that the jaccard overlap fits to a minimum value with the objects or randomly sampling a patch. Then each sampled patch is resized to a fixed size and flipped horizontally with probability 0.5 and in addition, applying some photo-metric distortions. During the model analysis of SSD it was showed that both implementing data augmentation and having more default box shapes are crucial to increase the model accuracy.

During inference is essential to perform non-maximum suppression to filter the large number of boxes generated. The results on PASCAL VOC2007 showed that with a low 300x300 resolution the SSD is more accurate than Fast R-CNN and when the resolution is larger (512x512), it surpasses Faster R-CNN by 1.7% mAP. Using data augmentation, the results showed than the SSD300 model already surpasses Faster R-CNN by 1.1% mAP. Compared to YOLO, SSD300 achieves better accuracy for smaller input images on VOC2007 test: 74.3% mAP at 59 FPS.
Compared to R-CNN, SDD has less localization error but more confusions with similar objects because SDD shares locations for multiple categories. SDD performs worse in small objects than in bigger ones and that can be solved increasing the input image size from 300x300 to 512x512, thus reducing the speed. 

## [Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/)
### Authors: Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, Antonio Torralba

### Summary
In this work, a technique to perform weakly supervised object localization is presented. It can be applied to any Convolutional Neural Network used for classification. By introducing a slight modification in the network, this can be used for object localization, despite being trained only for classification.

The main idea behind this technique is to remove the densely connected layers of the network, and apply a global average pooling after the last convolutional layer, followed by a dense layer and a softmax. The new network obtained this way may have a slightly lower accuracy, but it can be used to create the Class Activation Map (CAM) for each category. This CAM can be used then to localize objects.

In order to obtain the CAM, once the modified network has been trained, all the new layers are removed, and a 1x1 convolution is placed after the last convolution, with the weights of the dense layer that we used before. The output of such a network is a heatmap for each category, with higher values where the features associated with such category are found.

In the paper, this technique is applied to GoogLeNet and VGG-16. These networks are the used in ILSVRC to perform localization, reaching a top-5 error close to AlexNet, which is fully supervised (37.1% against 34.2). However, other networks that are also trained in a fully supervised way are better by a wider margin.

# Segmentation

## [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)
### Authors: Jonathan Long, Evan Shelhamer, Trevor Darrell

In this paper, the authors adapt the most used classification networks into fully convolutional networks (FCN) and fine-tune them to the segmentation task. These so-called fully convolutional networks exceed the state-of-the art in semantic segmentation, while simplifying and speeding up learning inference.

This network combines information from a coarse and a shallow layer to obtain accurate segmentations. The method adapts deep classification architectures, using image classification as supervised pre-training, and fine-tune fully convolutionally to learn the whole image inputs and ground truths. An FCN operates on an input of any size and produces an output of corresponding spatial dimensions. 

Fully connected layers can be seen as convolutions with kernels that cover their entire input regions, so they can take as input any image size and output a classification map. However, the output dimensions are reduced by subsampling.
The FCN model tried to implement a trick introduced in OverFeat to obtain dense predictions from coarse outputs without interpolation. It uses input shifting and output interlacing with the tradeoff that the output is made denser without decreasing the receptive field sizes of the filters but the filters are prohibited from accessing information at a finer scale than their original design.

Finally, they decided to implement learning through up sampling as it was more efficient when combined with the skip layer fusion.
Interpolation is another way to connect coarse outputs to dense pixels. A natural way to upsample is backwards convolution (deconvolution) with an output stride of f. They use in-network upsampling as a fast and effective way to learn dense predictions. 

![model fcn](https://raw.githubusercontent.com/sunshineatnoon/Paper-Collection/master/images/FCN1.png "FCN8")

#### Segmentation Architecture
The model casts ILSVRC classifiers into FCNs and augment them for dense prediction with in-network upsampling and a pixelwise loss. We train for segmentation by fine-tuning and use a novel skip architecture that combines coarse, semantic and local, appearance information to refine prediction. 

Optimization: they train the network with SGD momentum using a minibatch size of 20 images and fixed learning rates for each model used: FCN-AlexNet, FCN-VGG16 and FCN-GoogLeNet. The class scoring convolutions where zero-initialized and the Dropout layers of the base models kept.

Fine-tuning: all the layer are fine-tuned by backpropagation through the whole net. Fine-tuning takes three days on a single GPU for the coarse FCN-32s version and about one day for the FCN-16s and FCN-8s.

Patch sampling: the full image is divided into a regular grid of large, overlapping patches. Sampling the input images during training did not increase the performance and took more time.

Class Balancing: the classes can be balanced by sampling or weighting the loss but was not necessary to boost the performance.

Dense Prediction: Final layer deconvolutional filters are fixed to bilinear interpolation, while intermediate upsampling layers are initialized to bilinear upsampling and then learned.

Augmentation: data augmentation was not used due it did not increase the performance but adding more data using a much larger dataset improved the results by 3.4 points. 

#### Results
Four metrics were used from common semantic segmentation and scene parsing evaluations that are variations on pixel accuracy and region intersection over union. The pixel accuracy, the mean accuracy, the mean IU and the frequency weighted IU. 

## [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/pdf/1511.00561.pdf)
### Authors: Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla

In this paper, the authors presented an encoder-decoder engine for image semantic segmentation. The focus of this architecture is to obtain features which are useful for accurate boundary localization. They also considered the computational perspective looking for a network efficient in memory and computational time terms. Finally, they considered important that the network has the ability of being trained end-to-end.

The encoder network is composed by the 13 convolutional layers of VGG16, for this reason they e initialize the training process from weights trained for classification on large datasets. The decoder network consists of a hierarchy of decoders one corresponding to each encoder, decoders use the max-pooling indices received from the corresponding encoder to perform non-linear up sampling of their input feature maps. With this technique, the boundary delineation is improved and the number of parameters for training is reduced. The final decoder output is fed to a multi-class soft-max classifier to produce class probabilities for each pixel independently.

Each encoder performs a convolution, then a batch normalization, after that ReLU is applied, followed by a max-pooling with a 2x2 window and finally a subsampling with a 2 factor is applied. Max-pooling and subsampling achieve translation invariance for classification but on the other hand there is a loss of spatial resolution. To avoid this loss, the locations of the maximum feature value in each pooling window is memorized for each encoder feature map. The decoder up samples the input feature map using the memorized max-pooling indices from the corresponding encoder, then its output is convolved to obtain dense feature maps and finally batch normalization is applied.

They designed some different decoder variants:
- Bilinear-Interpolation: up sampling using fixed bilinear interpolation weights, it does not require learning for up sampling.
- SegNet-Basic: 4 encoders and 4 decoders. The encoders perform max-pooling (saving indices) and subsampling, batch normalization is applied both on the encoder and decoder, there are no baises and no ReLU nonlinearity in the decoder. A 7x7 kernel is applied both on the encoder and the decoder, getting a wide image context.
- SegNet-Basic-EncoderAddition: Use the pooling indices for up sampling followed by a convolutional step to densify the input and then add the corresponding 64 encoder feature maps to produce the decoders output.
- SegNet-Basic-SingleChannelDecoder: SegNet-Basic version where the decoder filters are single channel.
- FCN-Basic: SegNet-Basic encoder network and FCN decoding technique on the decoder network.
- FCN-Basic-NoAddition: FCN-Basic variant without the encoder feature map addition step and only learns the up-sampling kernels.
- FCN-Basic-NoDimReduction: There is no dimensionality reduction performed for the encoder feature maps, the number of channels at the end of each decoder is the same as the corresponding encoder.
- FCN-Basic-NoAddition-NoDimReduction: Variant without addition step and dimensionality reduction for the encoder feature maps.

For training, they used the CamVid dataset performing previously a local contrast normalization. They use stochastic gradient descent (SGD) with a fixed learning rate of 0.1 and momentum of 0.9, training until the cross-entropy loss convergence. The used mini-batch is composed by 12 images and before each epoch the training set is shuffled. They applied class balancing (weight the loss differently depending on the true class) applying the median frequency balancing algorithm, the weight is ratio of the median of class frequencies computed on the entire training set divided by the class frequency.

To compare the performance, they used:
- Global accuracy: Percentage of pixels correctly classified.
- Class average accuracy: mean of the predictive accuracy over all classes.
- Mean intersection over union: where false positive predictions are penalized.
- Boundary F1-measuer: average the image F1 measures (precision and recall values between the predicted and ground truth class boundary given a pixel tolerance distance).

Analyzing the obtained results, they had seen the importance of learning decoders for segmentation. They also see that SegNet and FCN-Basic performed similarly but SegNet uses less memory during inference because it only saves max-pooling indices, but FCN-Basic is faster because it has less convolutions on the decoder part. They also realize that SegNet-Basic outperforms FCN-Basic-NoAddition because it captures the information present in the encoder feature maps. Another observation is that SegNet-Basic has a competitive training accuracy when compared to larger models such FCNBasic-NoDimReduction. The comparison between FCN-BasicNoAddition and SegNet-Basic-SingleChannelDecoder shows that using max-pooling indices for up sampling and an overall larger decoder leads to better performance. When both memory and inference time is not constrained, larger models such as FCN-Basic-NoDimReduction and SegNet-EncoderAddition are both more accurate. Finally, they observed that without weighting the results are poorer for all the variants.
They concluded that the best performance is achieved when encoder feature maps are stored in full, that to improve performance when inference memory is constrained encoder feature maps can be compressed before storing and that larger decoders increase performance for a given encoder network.

They compared SegNet performance with other techniques. First, they compared with CRFs and realize that SegNet obtains competitive results, showing the ability of deep architectures to extract meaningful features from the input image and map it to accurate and smooth class segment labels. They also compared with state-of-the-art deep techniques and saw that DeconvNet has a higher boundary delineation accuracy but SegNet is much more efficient as compared to DeconvNet. For the FCN model learning the deconvolutional layers as opposed to fixing them with bi-linear interpolation weights improves performance particularly the BF score.

Finally, they studied the SegNet performance on indoor images and realized that SegNet obtains reasonable predictions when the size of the classes are large under different viewpoints. However as compared to outdoor scenes the segmentation quality is clearly noisier.
