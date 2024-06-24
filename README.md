# Knowledge Distillation of Convolutional Neural Networks

As a final project of this course, we used Knowledge Distillation (KD) to train a convolutional neural network so it can learn to classify the 10 instances of the CIFAR-10 dataset. 

A total of 4 different architectures were used, and the versions that used distillation were compared against those that did not. The results obtained showed that using knowledge distillation is more effective than training directly with the labels as it prevents overfitting. Finally, the pros and cons of using this methodology are discussed.

## Experiments

All models were trained for 50 epochs using SGD with a momentum of 0.9, as well as a scheduler to adjust the learning rate during training to avoid overfitting. Models that did not use distillation were trained with an initial learning rate of 0.01, while distillates used 0.1, temperature of 8 and a w_1=w_2=0.5. 

The following graph shows the total accuracies of each model.