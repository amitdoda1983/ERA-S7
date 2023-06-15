# ERA-S7

## Building custom CNN model to achieve 99.4 % accuracy in less than 15 epochs using less than 8000 parameters

### Experiment 1 : ERA1S7_Net1

#### targets : Build the right skeleton model which gets to required receptive field of 24/28 and 99.4% accuracy 

#### results : The model is able to achieve an accuracy of 99.39% in 21st epoch. Total params: 7,984

#### analysis : The skeleton with right number of layers is achieved however the convergence is slow.The model is also overfitting suggesting need for regularizers

links : https://github.com/amitdoda1983/ERA-S7/blob/main/ERA1S7_Net1.ipynb


### Experiment 2 : ERA1S7_Net2

#### targets : To achieve 99.4 in less than 15 epochs using less than 8k parameters

#### results: Achieved 99.43% in 13th epoch. Total params: 7,984

#### analysis :To solve the problem of slow convergence, reduceLRonplateau is used.Using LR scheduler helped achieve this accuracy once. The LRreduceonplateau helped achieve faster convergence.
with this i could start with a relatively higher lr in the start and could reduce the lr whenver the plateau hit.It greatly helped in achieving better accuracy in lesser epochs.
It allowed trimming the number of kernels in the network which helped reduce the number of parameters well within the 8000 range.I could just hit the 99.4 plus accuracy once and also observed some overfitting which needs to be fixed.So i need to do explore other techniques such as data augmentation and regularizer such as dropout.

links : https://github.com/amitdoda1983/ERA-S7/blob/main/ERA1S7_Net2.ipynb


### Experiment 3 : ERA1S7_Net3

#### targets : To consistently achieve 99.4 plus accuracy in 15 epochs.

#### results : Achieved 99.4 % plus consistently within 15 epochs. Got 99.43 in 12th, 99.47 % in 13th , 99.44 % in 14th and 99.43 % in 15th epoch.Total params: 7,984

#### analysis : Using dropout and data augmentation helped achieve the desired accuracy consistently. with this the model I was able to classify hard images also which were rotated or were prone to human error too.The gap between train and test accuracy is reduced giving us a model which does equally good on both.

links : https://github.com/amitdoda1983/ERA-S7/blob/main/ERA1S7_Net3.ipynb
