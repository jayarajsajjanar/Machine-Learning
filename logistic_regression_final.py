import cPickle
import gzip
import numpy as np
import datetime
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = cPickle.load(f)

X = np.matrix(training_data[0])

t_temp1 = np.matrix(training_data[1])#[[1,3,2....2]]
t_temp2 = t_temp1.T
#one hotting t
t = np.zeros((50000, 10))#one hot encoded form of t.
t[np.arange(50000),t_temp2.T] = 1

b = np.ones((1,50000))


w = np.zeros((10,784))

a = np.zeros((10,50000))

epoch = 0 
for epoch in range(100):

	for i in range(10):
		a[i] = w[i] * X.T + b

	#converting to a to prob 
	a_sum = a.sum(axis =0 )
	a_prob = a/a_sum

	y_temp = a_prob.argmax(axis=0) #y_temp is a row matrix.. doesnt have any columns .. hence modifying.
	y_temp2 = np.zeros((1,50000))
	y_temp2 = y_temp #[[0,0,0....0]] i.e. 1 row 50k cols.
	#one hot encoding y from y_temp2
	y = np.zeros((50000, 10))
	y[np.arange(50000),y_temp2] = 1

	w = w - 0.3 * (y-t).T * X

	if epoch%3==0:
		print "epoch#:{} at {} ".format(epoch, datetime.datetime.now()),
	 	print "current state of prediction i.e y : {}".format(y_temp2),

		# print np.sum(w)
		sse = np.sqrt(np.sum(np.asarray(y_temp2 - t_temp1)**2))
		print "sum of squared errors = {}".format(sse)
		if sse > 300:
			continue
		else:
			break
print "***total correctly predicted on TRAINING data:{}***".format((t_temp1==y_temp2).sum())
print "classification error rate on TRAINING data: {}".format(float(float((t_temp1!=y_temp2).sum())/50000))
#_____________________________________________________________________________________________
#checking on validation data after weights are obtained.
X_valid = np.matrix(validation_data[0])
t_temp1_valid = np.matrix(validation_data[1])#[[1,3,2....2]]
t_temp2_valid = t_temp1_valid.T
b_valid = np.ones((1,10000))
a_valid = np.zeros((10,10000))

for i in range(10):
		a_valid[i] = w[i] * X_valid.T + b_valid
a_valid_sum = a_valid.sum(axis =0 )
a_valid_prob = a_valid/a_valid_sum

y_valid_temp = a_valid_prob.argmax(axis=0)
y_valid_temp2 = np.zeros((1,10000))
y_valid_temp2 = y_valid_temp

print "***total correctly predicted on VALIDATION data:{}***".format((t_temp1_valid==y_valid_temp2).sum())
print "classification error rate on VALIDATION data: {}".format(float((t_temp1_valid!=y_valid_temp2).sum())/10000)
#_____________________________________________________________________________________________
#checking on test data after weights are obtained.
X_test = np.matrix(test_data[0])
t_temp1_test = np.matrix(test_data[1])#[[1,3,2....2]]
t_temp2_test = t_temp1_test.T
b_test = np.ones((1,10000))
a_test = np.zeros((10,10000))

for i in range(10):
		a_test[i] = w[i] * X_test.T + b_test
a_test_sum = a_test.sum(axis =0 )
a_test_prob = a_test/a_test_sum

y_test_temp = a_test_prob.argmax(axis=0)
y_test_temp2 = np.zeros((1,10000))
y_test_temp2 = y_test_temp

print "***total correctly predicted on TEST data:{}***".format((t_temp1_test==y_test_temp2).sum())
print "classification error rate on TEST data: {}".format(float((t_temp1_test!=y_test_temp2).sum())/10000)

#_____________________________________________________________________________________________
#checking on USPS data after weights are obtained.

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
			resized_image = cv2.resize(img, (20, 20))
			padded_image = cv2.copyMakeBorder(resized_image,4,4,4,4,cv2.BORDER_CONSTANT,value=[255,255,255])
			reshaped_image = np.reshape(padded_image, (1,784))
			x_usps[x_index] = reshaped_image
			y_usps[y_index] = i
			x_index+=1
			y_index+=1
	# print i,countt
print "predicting on USPS"

b_usps = np.ones((1,19999))
a_usps = np.zeros((10,19999))

for i in range(10):
		a_usps[i] = w[i] * x_usps.T + b_usps
a_usps_sum = a_usps.sum(axis =0 )
a_usps_prob = a_usps/a_usps_sum

y_usps_temp = a_usps_prob.argmax(axis=0)
y_usps_temp2 = np.zeros((1,19999))
y_usps_temp2 = y_usps_temp

# print "***total correctly predicted on TEST data:{}***".format((t_temp1_test==y_test_temp2).sum())
# print "classification error rate on TEST data: {}".format(float((t_temp1_test!=y_test_temp2).sum())/10000)

count = 0 
for i in range(len(y_usps)):
	if y_usps_temp2[i]!=y_usps[i]:
		count+=1
# print count
print "classification error rate:{}".format(float(count) / float(19999))