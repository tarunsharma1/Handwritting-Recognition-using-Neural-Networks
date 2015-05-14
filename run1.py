import pickle
import numpy as np
import mnist
import math
from PIL import Image
from numpy import array
import matplotlib.pyplot as plt


x1=784
y1=30
z1=10

theta1=pickle.load(open("theta1_new.npy","rb"))
theta2=pickle.load(open("theta2_new.npy","rb"))

images,labels = mnist.load_mnist('training')
a2=np.empty([y1+1,1],dtype='float64') #31x1
output=np.empty([z1,1],dtype='float64') #10x1

def g(z):
	return (1.0/(1.0+math.exp(-z)))

def feedforward2(input1):
	z2=np.dot(theta1,input1)
	z2=z2/100.0
	#print theta1
	for i in range (1,(y1+1)):
             a2[i] = g(z2[i-1])
	a2[0]=1.0
	z3=np.dot(theta2,a2)
	for i in range (0,z1):
             output[i] = g(z3[i])
	list1 = [a2,output]
	return list1;


#im = Image.open("two.jpg")
#image1 = array(im,dtype='f')
#image1 = image1.reshape(784,1)
image1=images[142].reshape(784,1)
input1=np.zeros([785,1],dtype='float64')
for i in range(1,785):
	input1[i]=image1[i-1]/(255.0)
input1[0]=1.0

c=feedforward2(input1);
index=np.where(c[1]==max(c[1]));
#print c[1];
print "\n Number is ",index[0][0];
plt.bar(np.arange(10),c[1],1);
plt.xlabel('Digits');
plt.ylabel('Probability of recognition');
plt.title('Results');
plt.xticks(np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]), ('0','1', '2', '3', '4', '5','6','7','8','9'))
plt.show();
