# EVA-8_Phase-1_Assignment-9
This is the assignment of 9th session in Phase-1 of EVA-8 from TSAI

## Introduction

### Objective
Objective of this assignment is to train a custom writtern single-head attention written in [cv wrapper repo](https://github.com/devdastl/eva8_source/blob/main/models/model_9.py) on [CIFAR10 Dataset](http://yann.lecun.com/exdb/mnist/) and should adhare to following conditions:
1. Should have three convulation layer to bring 32x32x3 to 1x1x48. Should use GAP layer to get 1x1.
2. Write Ultimus block to perform Attenion matrxi calculation and repeat it 4 times.
3. Finally generate (batch_size, 10) for loss claculation.
4. Use Adam optimizer and One-Cycle-Policy to train the model for 24 epochs.

### Getting started
It is very easy to get started with this assignment, just follow below mentioned steps:
1. Open assignment 9 notebook in google colab.
2. Run first cell to clone auxelary repo into the current runtime of colab. Please note that this repo has been upgraded to support modulerised training loop.
4. Note that deleting runtime can reset and delete cloned repository.

## About Attention Mechanism
Attention is a mechanism that allows the Transformer model to focus on different parts of the input sequence during training and inference. The Transformer architecture is a type of neural network that is specifically designed for natural language processing (NLP) tasks, such as language translation, question answering, and text summarization.

In the Transformer architecture, attention is used to calculate a weight for each input token, which represents its importance for predicting the output token. The attention weights are computed based on the similarity between the input token and a set of learned query, key, and value vectors. The query vector is used to determine which parts of the input sequence to attend to, while the key and value vectors are used to represent the input tokens and their associated values.

There are two type of Attention Mechanism:
1. **Self-attention:** Self-attention is a mechanism that allows an entity to attend to its own internal features or context in order to make predictions or decisions. It is commonly used in natural language processing tasks such as machine translation and text summarization.

In other word in self-attention Key, Query and Value comes from same data distribution.

2. **Cross-attention:** cross-attention is a variant of self-attention where the entity attends to features or context from another entity or source, such as attending to the source sentence in machine translation or attending to the image in image captioning. Cross-attention is often used to improve the performance of models that deal with multiple modalities or sources of information.

In other word Query comes from other distribution compare to Key and Value in cross-attention.
Self-Attentention                    | Cross-Attention
:---------------------------------------------------:|:--------------------------------------------------:
![Alt text](report/self_attention.jpg?raw=true "")  | ![Alt text](report/cross_attention.png?raw=true "")

## Data representation
In this assignment I am using [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) with this dataset I am applying following augmentation on the top:
1. `RandomCrop` - Cropping 32x32 patches from the input image after giving a padding of 4.
1. `HorizontalFlip` - Fliping the image along horizontal axis.
3. `CoarseDropOut` - Overlay a rectangle patch(half the size of original image) on a image randomly. (simulate object hindarence)
6. `Normalize` - Normalize image i.e. zero centring (zero mean) and scaling (one std)

Below is the graph representing the input training dataset after appling all augmentations.
![Alt text](report/data_6.png?raw=true "model architecture")

## Model representation
For this assignment we written a single-head self attention block repeated 4 times followed by 3 convolution layers.
Below image shows architecture of this attention block:

![Alt text](report/model_arch.png?raw=true "model architecture")

Now let's see the output of model summary for the above mentioned architecture. Below is the log of torch_summary output.
```
-------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 30, 30]             432
            Conv2d-2           [-1, 32, 28, 28]           4,608
            Conv2d-3           [-1, 48, 26, 26]          13,824
 AdaptiveAvgPool2d-4             [-1, 48, 1, 1]               0
            Linear-5                    [-1, 8]             384
            Linear-6                    [-1, 8]             384
            Linear-7                    [-1, 8]             384
            Linear-8                   [-1, 48]             384
           Ultimus-9                   [-1, 48]               0
           Linear-10                    [-1, 8]             384
           Linear-11                    [-1, 8]             384
           Linear-12                    [-1, 8]             384
           Linear-13                   [-1, 48]             384
          Ultimus-14                   [-1, 48]               0
           Linear-15                    [-1, 8]             384
           Linear-16                    [-1, 8]             384
           Linear-17                    [-1, 8]             384
           Linear-18                   [-1, 48]             384
          Ultimus-19                   [-1, 48]               0
           Linear-20                    [-1, 8]             384
           Linear-21                    [-1, 8]             384
           Linear-22                    [-1, 8]             384
           Linear-23                   [-1, 48]             384
          Ultimus-24                   [-1, 48]               0
           Linear-25                   [-1, 10]             480
================================================================
Total params: 25,488
Trainable params: 25,488
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.55
Params size (MB): 0.10
Estimated Total Size (MB): 0.66
-------------------------------------------------------------

```
In above log we have 4 instances of Ultimus block and each block has 4 linear layer. 3 of them represent Query, Key and Value scaled from 48 to 8 and the fourth one is used to scale back the output from 8 to 48 dimension.

## Training log
Below are the last five epoch training logs for single-head attentinon model which has been trained for 24 epoch using One Cycle Policy.

```
EPOCH: 19
Loss=15.465812683105469 Batch_id=97 Accuracy=11.39: 100%|██████████| 98/98 [00:12<00:00,  7.73it/s]
Test set: Average loss: 0.0476, Accuracy: 1317/10000 (13.17%)

EPOCH: 20
Loss=136.10498046875 Batch_id=97 Accuracy=11.66: 100%|██████████| 98/98 [00:12<00:00,  7.81it/s]
Test set: Average loss: 0.3257, Accuracy: 1462/10000 (14.62%)

EPOCH: 21
Loss=18.676441192626953 Batch_id=97 Accuracy=12.47: 100%|██████████| 98/98 [00:12<00:00,  7.82it/s]
Test set: Average loss: 0.0332, Accuracy: 1347/10000 (13.47%)

EPOCH: 22
Loss=6.530598163604736 Batch_id=97 Accuracy=12.58: 100%|██████████| 98/98 [00:12<00:00,  7.72it/s]
Test set: Average loss: 0.0211, Accuracy: 1332/10000 (13.32%)

EPOCH: 23
Loss=9.75550651550293 Batch_id=97 Accuracy=12.66: 100%|██████████| 98/98 [00:12<00:00,  7.74it/s]
Test set: Average loss: 0.0184, Accuracy: 1286/10000 (12.86%)

EPOCH: 24
Loss=7.363772392272949 Batch_id=97 Accuracy=12.73: 100%|██████████| 98/98 [00:12<00:00,  7.80it/s]
generating mis-classified images for epoch 24
generating mis-classified images for epoch 24
generating mis-classified images for epoch 24
generating mis-classified images for epoch 24
generating mis-classified images for epoch 24
generating mis-classified images for epoch 24
generating mis-classified images for epoch 24
generating mis-classified images for epoch 24
generating mis-classified images for epoch 24
generating mis-classified images for epoch 24

Test set: Average loss: 0.0157, Accuracy: 1299/10000 (12.99%)
```

## Results
One this to notice here is we didn't use multple heads as well as we didn't divide our image into patches. Main motive of this assignment is to get familiarized with writing attention blocks for the transformer.

Below are the generated graphs for training and evaluation of the model:
Accuracy-Loss graph for training                     | Accuracy-Loss graph for validation
:---------------------------------------------------:|:--------------------------------------------------:
![Alt text](report/graph_train_9.png?raw=true "")  | ![Alt text](report/graph_eval_9.png?raw=true "")
