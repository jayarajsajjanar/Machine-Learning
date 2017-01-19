from os import listdir
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage
import scipy as sp
import numpy as np
import pickle
print "loading USPS"
x_usps = np.zeros((19999,784))
x_index= 0
y_index= 0
y_usps = np.zeros((19999,1))
for i in range(10):
	# print "folder {}".format(i)
	loc = '/media/23e97447-9196-4c8e-be28-b24a49d2548e/home/jayaraj/Study/CS_ML/project3/USPSdata/Numerals/'+str(i)+'/'
	countt = 0 
	for f in listdir(loc):
		if f[-3:-1] == 'pn':
			countt+=1
			file_name = loc+f
			img = sp.ndimage.imread(file_name,flatten=True )
			resized_image = cv2.resize(img, (20, 20))
			padded_image = cv2.copyMakeBorder(resized_image,8,8,8,8,cv2.BORDER_CONSTANT,value=[255,255,255])
			reshaped_image = np.reshape(padded_image, (1,784))
			x_usps[x_index] = reshaped_image
			y_usps[y_index] = i
			x_index+=1
			y_index+=1
	# print i,countt
# print "predicting on USPS"
# b1_usps = np.ones((19999,784))
# b2_usps = np.ones((19999,10))

# z1_usps = np.matrix(x_usps/255) * w1.T + b1_usps
# a1_usps = 1.0 / (1.0 + math.e ** (-1.0 * np.array(z1_usps)))
# z2_usps = np.matrix(a1_usps) * w2.T + b2_usps

# a2_sum_usps = np.array(z2_usps).sum(axis =1 )
# a2_prob_usps = np.array(z2_usps)/a2_sum_usps[0]

# a2_max_probs_usps = a2_prob_usps.argmax(axis = 1 )

# total_correct_usps = (np.array(y_usps)==a2_max_probs_usps).sum()
# print "total correct usps={}".format(total_correct_usps)

with open('x_usps.pickle', 'wb') as handle:
    pickle.dump(x_usps, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('y_usps.pickle', 'wb') as handle:
    pickle.dump(y_usps, handle, protocol=pickle.HIGHEST_PROTOCOL)

f1 = open('x_usps.pickle', 'rb')
c = pickle.load(f1)
print c

f2 = open('y_usps.pickle', 'rb')
d = pickle.load(f2)
print d