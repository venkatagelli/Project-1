###### 1.   How many layers

We need to add as many layers as required to "see" the whole image because network may not be able to figure out the object just from a partial view

Our network should read all edges and gradients to make textures. Then it should make patterns from these. Then it should make parts of objects. From parts of objects, it should make objects. From objects it should make scene. Hence layers are used everywhere. 

We can keep adding layers as long as there is no over fitting or there is no improvement in test error



###### 2.   Kernels and how do we decide the number of kernels?

Kernels are also known as filters. The kernels are used to perform convolution on input image. We can think number of kernels as hyper-parameter. generally we consider 

Inter and intra class variations 
Expressivity required 
Hardware capacity 

while deciding the number of kernels.

###### 3.  3x3 Convolutions

3x3 Kernels are highly optimized.

3x3 Kernels add fewer number of parameters i.e 9 when compared with higher order Kernels. 

Two 3x3 kernels will add 18 parameters to the network and is equivalent to one 5x5 kernel where as A 5x5 Kernel will add 25 parameters. 

###### 4.  Receptive Field

The receptive field in a convolutional neural network refers to the part of the image that is visible to one filter at a time. Generally global receptive field refers to the size of the input image.



###### 5.  Batch Normalization

Batch normalization is a technique for improving the speed, performance, and stability of artificial neural networks. It is used to normalize the input layer by adjusting and scaling the activations. Batch normalization smoothens the objective function to improve the performance. Batch normalization achieves length-direction decoupling, and thereby accelerates neural networks.

It can be used before or after the convolution

It should be added before prediction layer

###### 6.  The distance of Batch Normalization from Prediction

Batch Normalization must be used at least 2-3 layers before prediction layer

###### 7.  Image Normalization

Image Normalization is a preprocessing step used to bring uniformity to the input images.

###### 8.  MaxPooling

Maxpooling is used to reduce the resolution of the output. Generally wee prefer 2x2 maxpooling. 2X2 maxpooling will reduce the output resolution to half.

Small local positional invariance, small scale / skew invariance, small rotational invariance are some benefits of maxpooling.

###### 9.  Position of MaxPooling

Generally used after edges and gradients are formed,  Textures are formed, patterns, are formed, parts of objects and objects are formed.

It shouldn't be used before final convolution layer.

###### 10   The distance of MaxPooling from Prediction

Maxpooling must be used at least 2-3 layers before prediction layer

###### 11.   Concept of Transition Layers

To extract the required features and to combine the channels we use transition layers

###### 12.   Position of Transition Layer

These are used after convolution block

###### 13.   DropOut

Dropout is a simple regularization model. It can be used to reduce the overfitting

###### 14.   When do we introduce DropOut, or when do we know we have some overfitting

If our network is performing well with training data and and not doing well with test data then we know that there is a overfitting issue.



###### 15.   1x1 Convolutions

1x1 convolution is used for the below purposes.

Less number of parameters 
Lesser computation requirement for reducing the number of channels
Use of existing channels to create complex channels (instead of re-convolution)
We can use 1x1 to increase the number of channels, just that we need to have a purpose

###### 16.   SoftMax

Softmax is an activation function to create distance / seperation between classes.
Its not probability. Its probability like.
Its not actual value. Its used to keep us happy.
When using softmax in medical / critical / life related fields, we need to check the top1, top2, top3 classes also.
Its a pinch of salt.

###### 17.   Adam vs SGD

SGD is a variant of gradient descent. Instead of performing computations on the whole dataset — which is redundant and inefficient — SGD only computes on a small subset or random selection of data examples. SGD produces the same performance as regular gradient descent when the learning rate is low.

Essentially Adam is an algorithm for gradient-based optimization of stochastic objective functions. It combines the advantages of two SGD extensions — Root Mean Square Propagation (RMSProp) and Adaptive Gradient Algorithm (AdaGrad) — and computes individual adaptive learning rates for different parameters.

###### 18.   Learning Rate

The learning rate is a configurable hyperparameter used in the training of neural networks that has a small positive value, often in the range between 0.0 and 1.0

###### 19.   Number of Epochs and when to increase them

Number of epochs multiplyed by batch size give the total number of input images. If the batch size is small we increase the number of epoch.

Epoch is number of iteration related with each input samples in the dataset. Large value of epoch leads to train more times which sometimes gives better accur



###### 20.   How do we know our network is not going well, comparatively, very early

Compare current networks first 3-4 epoch runs with previous networks first 3-4 epoch runs. If the latest network is not doing good, then we say its not going well

###### 21.   Batch Size, and effects of batch size

Batch size depends on hardware constraints. Bigger the batch size, less the execution time of epoch.  Bigger batches will give better accuracy.

###### 22.   LR schedule and concept behind it

When training deep neural networks, it is often useful to reduce learning rate as the training progresses. This can be done by using pre-defined learning rate schedules or adaptive learning rate methods. Learning rate schedules seek to adjust the learning rate during training by reducing the learning rate according to a pre-defined schedule. 

###### 23.   When to add validation checks

###### 24.   When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)

