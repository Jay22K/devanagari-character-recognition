# Devanagari-Character-Recognition
I have used convolutional neural networks  for create the model. 

Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning.
For this project you need to install Keras , Tensorflow , Numpy Libraries. 

Python Implementation
1. Dataset- DHCD (Devnagari Character Dataset)
2. Images of size 32 X 32
3. Convolutional Network Support added.

I have upload the model. Architecture of this model is given below  

Architecture :   CONV2D -->CONV2D--> MAXPOOL --> CONV2D-->CONV2D --> MAXPOOL -->DROPOUT-->FLATTEN-->FC -->Softmax  

Accuracy of this model is ~98%  

In this project I have upload the python code that will recognise the characters ad digits of Devanagari Lipi   

Description : This code successfully recognises hindi characters and digits. (in future this code recognises Modi Lipi also)

- A convolutional neural network (CNN or ConvNet) is a network architecture for deep learning that learns directly from data. CNNs are particularly useful for finding patterns in images to recognize objects, classes, and categories. They can also be quite effective for classifying audio, time-series, and signal data.

- A CNN is a kind of network architecture for deep learning algorithms and is specifically used for image recognition and tasks that involve the processing of pixel data. There are other types of neural networks in deep learning, but for identifying and recognizing objects, CNNs are the network architecture of choice


<img src="90650dnn2.jpeg">

- A CNN's architecture is analogous to the connectivity pattern of the human brain. Just like the brain consists of billions of neurons, CNNs also have neurons arranged in a specific way. In fact, a CNN's neurons are arranged like the brain's frontal lobe, the area responsible for processing visual stimuli. This arrangement ensures that the entire visual field is covered, thus avoiding the piecemeal image processing problem of traditional neural networks, which must be fed images in reduced-resolution pieces. Compared to the older networks, a CNN delivers better performance with image inputs, and also with speech or audio signal inputs.
