import cPickle
import gzip
import numpy as np
import time
import math 
from datetime import datetime
print "beggining:{}".format((datetime.now()))
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = cPickle.load(f)

print "data loaded"
#_____________________________________________________________________________________________
x = np.array(training_data[0])
#_____________________________________________________________________________________________
t_hot = np.zeros((50000, 10))#one hot encoded form of t.
t_hot[np.arange(50000),np.array(training_data[1])] = 1
#_____________________________________________________________________________________________

np.random.seed(1)
w1 = 2*np.random.random((784,784)) - 1
w2 = 2*np.random.random((10,784)) - 1
#_____________________________________________________________________________________________
b1 = np.ones((50000,784))
b2 = np.ones((50000,10))

#_____________________________________________________________________________________________

print "matrices initialised"
count = 0 

num_iters = 35 #*****************

print "number of iterations are set as :{}".format(num_iters)
print "Usually, 88% accuracy is achieved in 5 mins i.e.\
 35 iterations. Takes about 15 mins i.e. 103 iterations to achieve 90% accuracy"

#---looping 
while(1):

	print str(datetime.now()),

	count +=1
	print "Iteration#:{}".format(count),
	#_________________________________________________________________________________________
	#forward
	# z1 = x * w1.T + b1 
	z1 = np.matrix(x) * w1.T + b1 
	a1 = 1.0 / (1.0 + math.e ** (-1.0 * np.array(z1)))
	z2 = np.matrix(a1) * w2.T + b2

	a2_sum = np.array(z2).sum(axis =1 )
	a2_prob = np.array(z2)/a2_sum[0]

	a2_max_probs = a2_prob.argmax(axis = 1 )
	print "first 10 predicted values:{}".format(a2_max_probs[0:10]),
	y_hot = np.zeros((50000, 10))
	y_hot[np.arange(50000),a2_max_probs] = 1

	#_________________________________________________________________________________________
	#backward
	dk = y_hot - t_hot
	dj = np.multiply(np.multiply(np.matrix(dk) * w2,a1),1-a1)

	gradw2 = dk.T * z1
	gradw1 = dj.T * x

	w2 = w2 - 0.55 *1/50000* gradw2
	w1 = w1 - 0.55 *1/50000* gradw1

	total_correct = (np.array(training_data[1])==a2_max_probs).sum()
	print "Accuracy on train={}".format(float(total_correct)*float(100)/float(50000))


	# if total_correct>=45000:
	if count>36:
		break

	# print "sleeping"
	# time.sleep(5)

print "Final classification error rate training:{}".format(float(50000-total_correct) / float(50000))

del(b1)
del(b2)

#_____________________________________________________________________________________________
#checking on validation data after weights are obtained.
b1_valid = np.ones((10000,784))
b2_valid = np.ones((10000,10))


z1_valid = np.matrix(validation_data[0]) * w1.T + b1_valid
a1_valid = 1.0 / (1.0 + math.e ** (-1.0 * np.array(z1_valid)))
z2_valid = np.matrix(a1_valid) * w2.T + b2_valid

a2_sum_valid = np.array(z2_valid).sum(axis =1 )
a2_prob_valid = np.array(z2_valid)/a2_sum_valid[0]

a2_max_probs_valid = a2_prob_valid.argmax(axis = 1 )

total_correct_validation = (np.array(validation_data[1])==a2_max_probs_valid).sum()
print "total correct validation={}".format(total_correct_validation)
print "classification error rate validation:{}".format(float(10000-total_correct_validation) / float(10000))
del (z1_valid,a1_valid,z2_valid,a2_sum_valid,a2_prob_valid,a2_max_probs_valid)

#_____________________________________________________________________________________________
#checking on test data after weights are obtained.
b1_test = np.ones((10000,784))
b2_test = np.ones((10000,10))

z1_test = np.matrix(test_data[0]).dot(w1.T) + b1_test
a1_test = 1.0 / (1.0 + math.e ** (-1.0 * np.array(z1_test)))

z2_test = np.matrix(a1_test) * w2.T + b2_test
# a2_test = 1.0 / (1.0 + math.e ** (-1.0 * np.array(z2_test)))
a2_sum_test = np.array(np.exp(z2_test)).sum(axis =1 )
a2_prob_test = np.array(z2_test)/a2_sum_test[0]

a2_max_probs_test = a2_prob_test.argmax(axis = 1 )

total_correct_test = (np.array(test_data[1])==a2_max_probs_test).sum()
print "total correct test={}".format(total_correct_test)
print "classification error rate test:{}".format(float(10000-total_correct_test) / float(10000))

# del(z1_test,a1_test,z2_test,a2_sum_test,a2_prob_test,a2_max_probs_test)
#_____________________________________________________________________________________________

# print "end:{}".format((datetime.now()))

#_____________________________________________________________________________________________
#Testing on USPS data.
from os import listdir
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage
import scipy as sp
print "Loading USPS from folders.....(takes about 2 mins)"
x_usps = np.zeros((19999,784))
x_index= 0
y_index= 0
y_usps = np.zeros((19999,1))
for i in range(10):
	# print "folder {}".format(i)
	loc = './USPSdata/Numerals/'+str(i)+'/'
	countt = 0 
	for f in listdir(loc):
		if f[-3:-1] == 'pn':
			countt+=1
			file_name = loc+f
			img = sp.ndimage.imread(file_name,flatten=True )
			resized_image = cv2.resize(img, (28, 28))
			reshaped_image = np.reshape(resized_image, (1,784))
			x_usps[x_index] = reshaped_image
			y_usps[y_index] = i
			x_index+=1
			y_index+=1
	# print i,countt
print "predicting on USPS"
b1_usps = np.ones((19999,784))
b2_usps = np.ones((19999,10))

z1_usps = np.matrix(x_usps/255) * w1.T + b1_usps
a1_usps = 1.0 / (1.0 + math.e ** (-1.0 * np.array(z1_usps)))

z2_usps = np.matrix(a1_usps) * w2.T + b2_usps

a2_sum_usps = np.array(np.exp(z2_usps)).sum(axis =1 )
# a2_sum_usps = np.array(z2_usps).sum(axis =1 )
a2_prob_usps = np.array(z2_usps)/a2_sum_usps[0]

a2_max_probs_usps = a2_prob_usps.argmax(axis = 1 )

# total_correct_usps = (np.array(y_usps)==a2_max_probs_usps).sum()
# print "total correct usps={}".format(total_correct_usps)
count = 0 
for i in range(len(y_usps)):
	if y_usps[i]!=a2_max_probs_usps[i]:
		count+=1
# print count
print "classification error rate:{}".format(float(count) / float(19999))

