# Plantain-Shelf-life using CNN Architechture
Plantain Shelf life and there freshness parameters determining using machine learning 

Introduction and Pre-processing: 

 The dataset contains 42days of recorded.data.There are 27 biochemical features,a column specifying which day it is and 3 images of the plantain on that day;the.first two images are PNG files of the same plantain taken from different orientations, while the third one is a PDF of one of these images.The third image is hence not  used to train the model. Since the problem boils down to a multiclass imageclassification task,a CNN based approach is taken.A major challenge that the dataset poses is that there are 42 days of data i.e.,42 classes and only 2 images.perday.
 i.e.,per class.This isnot enough data per class to  train a decisive model with good performance.So,the data has been restructured. The data has been grouped based on the ‘Absence of Shrinkage’ parameter for it satisfactorily identifies an old plantain from a new one.
We use numpy,cv2,os, random modules to analize and convert the data from image to array. 

 Google's Tensorflow neural network is used for data monitoring, Data analysing and Data tracking. The high level API of Tensorflow - 'KERAS' is used  to train the model for high accuracy.

Architecture of CNN:


To resemble the distinguision of plantain based on their freshness parameters, a CNN model is trained and tested, relying upon  encoding of  labels. For this reason, the label is encoded based  upon the shelf life of plantain, discerning between 38 classes relied on the diffent day samples. Presuming we validate the freshness of plantain as a binary classification problem, where a plantain is provided the label 1 if it is similar to class sample and 0 in the case that the plaintain is unidentical with that class sample. Thus we can write- 
f(𝜃) = {1   𝑖𝑓 𝑥 ≤ 𝜃
              0   𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒}
Here, 𝜃 refers the hyperparameters of the classifier f(𝜃).
This convolutional neural network was trained and tested based on acoustic data of approximately 160 measures. In this the time signal was cropped and moved before apperentaing a Fourier transform. To expand the robustness of the model’s performance, data augmentation was applied by flipping  amplitude and phase spectrum both. In addition, to reduce the assosiation in between training and testing set it was taken into calculation that a plantain doesn't appear two times both in training and testing. 
The CNN is an architecture where a pooling layer is following two successive convolutional layers. 

 The adjoining figure details the architecture of the model and the number of parameters used.
 The total number of parameter is used is  173,549. This makes the model optimally complex for our small dataset,relatively faster and lightweight.

The above model is trained for keras augmentations for 550 Epochs on the Adam optimizer  Using Sparse Categorical CrossEntropy Loss function and "accuracy" as performance matrix.


Results: 

The model has decisively achived 97.69% accuracy on the test set. The performance model of the test set is summarised in the adjoining figure.

Results : Test set vs training set

Loss is a summation regarding the errors occured for each sample in training or validation set.  Loss is used in the training process for finding the "best" parametric values for the model. During the training process the objective is to minimize this value and  ultimately mimimize the value tends to zero after 30-35 epochs when trained over 550 epochs. The  graph of model-loss vs loss for test & train set is adjoined.
 
Accuracy is methodology to measure a  performance of classification model. It is  basically expressed as a percentage.  Accuracy is counting of  predictions , the predicted value is equal to the real value.  It's binary (false/true) for a distinct sample.  Accuracy is graphed  during the training period though the value associated with the overall model accuracy.  The graph of model-accuracy vs accuracy for test & train set is adjoined.

 

Discussion:

During hyperparameter optimization obvercoming the loss of model generalizability a dropout layer is introduced with rate equal to 0.1.

We  take flattern layer to convert multi dimensional input tensors into one dimension.

As taking more dense layer decrease the accuracy, we only take one dense layer and as it is the last or final layer we activate this layer using "Softmax" Activation function.

Our investigation suggests that the limited amount of data is the main problem. We have only 6 images per class on average and all those images are of the same plantain. A deep learning model needs to learn from examples with ample amount of variation in them. Learning from multiple images of the same object will not make a model more general, on the contrary it will make for a specialised model that learns to identify the different states of only that particular object which it has seen while training.

Our recommendation for a model that predicts the day range and mean biocemical features, is that,at least 3 different plantains be studied at the same time and pictures from various orientations(not just two) be captured. That way we will have at least 6 images per daylif only 2 orientations are considered) instead of 2. A bit more data will help us develop a more general model that is optimally complex. For a model that predicts the exact day an d other features, approximately 45-60 images are required per class; if there are 42 classes(as in here) then at least 42x45 i.e., 1890 images are rquired.
