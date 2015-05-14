import numpy as np
import mnist
import math
from random import randint
import pickle
import random

images,labels = mnist.load_mnist('training')
#print labels[550][0]
#print images[550].reshape(784,1)
x1=784
y1=30
z1=10
theta1 = np.random.random_sample((y1,(x1+1)))
theta2 = np.random.random_sample((z1,(y1+1)))
#theta1 = np.ones([y1,x1+1],dtype='float64')
#theta2 = np.ones([z1,y1+1],dtype='float64')
alpha = 3.0
#m=200

a2=np.empty([(y1+1),1],dtype='float64') #31x1
output=np.empty([z1,1],dtype='float64') #10x1

def g(z):
	return (1.0/(1.0+math.exp(-z)))
	
def feedforward(trial_input):
	z2=np.dot(theta1,trial_input)
	z2=z2/100.0
	for i in range (1,(y1+1)):
             a2[i] = g(z2[i-1])
	a2[0]=1.0
	z3=np.dot(theta2,a2)
	for i in range (0,z1):
             output[i] = g(z3[i])
	list1 = [a2,output]
	return list1;
#image1=images[0].reshape(784,1)
#input1=np.empty([785,1],dtype='float64')
#for i in range(1,785):
#	input1[i]=image1[i-1]
#input1[0]=1.0
#a=feedforward(input1);
#print a[1]

######--------------------------------------------------#####################
def backprop():
	global theta1
	global theta2
	global images
	global labels
	for l in range(0,30):
		for j in range(0,5000): #number of iterations for gradient descent
	##loop from 1 to 1000
			D1 = np.zeros((y1,(x1+1)))
			D2 = np.zeros((z1,(y1+1)))
			x=np.empty([785,1],dtype='float64')
			lower=10*j
	        	upper=lower+10
			for i in range(lower,upper):                                      #for every input
				temp1=images[i].reshape(784,1)
				temp1=temp1/(255.0)
				for k in range(1,(x1+1)):
					x[k]=temp1[k-1]
				x[0]=1.0
		
			## format output y
				temp2 = labels[i][0]
				if(temp2 == 0):
					y=np.array([[1.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
				elif(temp2 == 1):
					y=np.array([[0.0],[1.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
				elif(temp2 == 2):
					y=np.array([[0.0],[0.0],[1.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
				elif(temp2 == 3):
					y=np.array([[0.0],[0.0],[0.0],[1.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
				elif(temp2 == 4):
					y=np.array([[0.0],[0.0],[0.0],[0.0],[1.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
				elif(temp2 == 5):
					y=np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[1.0],[0.0],[0.0],[0.0],[0.0]])
				elif(temp2 == 6):
					y=np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[1.0],[0.0],[0.0],[0.0]])
				elif(temp2 == 7):
					y=np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[1.0],[0.0],[0.0]])
				elif(temp2 == 8):
					y=np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[1.0],[0.0]])
				else:
					y=np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[1.0]])
		
		
		#preprocessing done correctly for each input
				list1 = feedforward(x)
				a2 = list1[0];
				output = list1[1];
		#now calculate deltas
				delta3 = output - y
				ones = np.ones([(y1+1),1])
				temp3 = ones - a2
 				temp4 = np.empty(((y1+1),1),dtype='float64')
				for k in range(0,(y1+1)):
					temp4[k] = a2[k] * temp3[k]
				temp5 = np.dot(np.transpose(theta2),delta3)
				temp6 = np.empty(((y1+1),1),dtype='float64')
				for k in range(0,(y1+1)):
					temp6[k] = temp5[k] * temp4[k]
				delta2 = temp6
		##--------not sure of this-----
				delta2_new = np.empty((y1,1),dtype='float64')
		#because of dimension problem in D later, I am removing the first row of delta2 which is always zero.
		#this sort of makes sense because delta2[0] represents error of bias unit.
				for k in range(0,y1):
					delta2_new[k]=delta2[k+1]
		
		##--------------------------------

		# now for each input, we have calculated errors delta3 and delta2..we need to get the sums of errors for all inputs. D1 and 
                #  D2 for layers respectively.
				D1=D1 + np.dot(delta2_new,np.transpose(x))
				D2=D2 + np.dot(delta3,np.transpose(a2))
		
	#outside loop..now we have sum of errors as D1 and D2 and we use this in gradient descent to update theta values
				
			temp_matrix1 = theta1 - (((alpha)/(10.0))*D1)      
			temp_matrix2 = theta2 - (((alpha)/(10.0))*D2)
			theta1 = temp_matrix1
			theta2 = temp_matrix2
	
	training_data=zip(images,labels)	
	random.shuffle(training_data)
	images,labels = zip(*training_data)	
   
backprop();
#print theta1
pickle.dump(theta1,open("theta1_new.npy","wb"))
pickle.dump(theta2,open("theta2_new.npy","wb"))
image1=images[4].reshape(784,1)
image2=images[5].reshape(784,1)
input1=np.empty([785,1],dtype='float64')
input2=np.empty([785,1],dtype='float64')
for i in range(1,785):
	input1[i]=image1[i-1]/255.0
	input2[i]=image2[i-1]/255.0

input1[0]=1.0
input2[0]=1.0
c=feedforward(input1);
print c[1]
c1=feedforward(input2);
print c1[1]

