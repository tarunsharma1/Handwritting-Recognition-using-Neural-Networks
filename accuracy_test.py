import pickle
import numpy as np
import mnist
import math
from PIL import Image
from numpy import array


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

count=0
input1=np.zeros([785,1],dtype='float64')
for k in range (50000,60000):
	image1=images[k].reshape(784,1)
	for i in range(1,785):
		input1[i]=image1[i-1]/(255.0)
	input1[0]=1.0

	c=feedforward2(input1);
	b=np.where(c[1] == (max(c[1])))
	if b[0] == labels[k]:
		count=count+1
print count
