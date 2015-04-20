# Handwritting-Recognition-using-Neural-Networks
Handwritting recognition using a three layer neural network. Used MNIST dataset for training. Hidden layer contains 30 nodes. 
Used Andrew NGs Coursera video lectures on Neural Networks for understanding and implementation - https://class.coursera.org/ml-005/lecture
Also used this online book for reference - http://neuralnetworksanddeeplearning.com/chap1.html

Stochastic Gradient descend is used with learning rate of 3.0 .Fifety thousand images from MNIST were used for training and obtaining the weights.
The weights are then stored using Pickle in the files theta1.npy and theta2.npy respectively.
Training data is divided in batches of size 10. Only one epoch is run here, that is the entire 50,000 images are used for training
in batches of size 10. After a batch, a step of gradient descent is taken. The entire dataset is parsed through only once and 
hence only one epoch. 

mnist.py is used to load images from the mnist dataset. The dataset can be found here - http://yann.lecun.com/exdb/mnist/

train-images-idx3-ubyte.gz and train-labels-idx1-ubyte.gz are to be extracted where the program files are.

3 layer network - First layer contains 784 nodes (one node for each pixel, images in MNIST are 28x28 pixels in size).
Second layer contains 30 nodes(as mentioned in http://neuralnetworksanddeeplearning.com/chap1.html , a hidden layer containing 100 
nodes will provide a better accuracy). Output layer contains 10 nodes, one for each digit (gives probability of that digit).


Note: In the feedforward method in run1.py and handwrittingtrial1.py , z2=z2/100.0 to normailize the data. If not done,
the value of z2 will be large and hence the value of a2 will be all ones (due to sigmoid function).

With running only one epoch, the accuracy is 93.9%. The accuracy will most defintly increase with number of epochs and also number of
nodes in hidden layer.
