# coding=utf-8
# libraries & packages
import numpy
import math
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from os import listdir
from os.path import isfile, join

# this function is provided from the official site
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# from PIL import Image
# def ndarray2image (arr_data, image_fn):
# 	img = Image.fromarray(arr_data, 'RGB')
# 	img.save(image_fn)

from scipy.misc import imsave
def ndarray2image (arr_data, image_fn):
	imsave(image_fn, arr_data)

# set dataset path
dataset_path = './cifar_10/'

# define the information of images which can be obtained from official website
height, width, dim = #請填入影像的長寬與維度
classes = #請填入影像類別個數

''' read training data '''
# get the file names which start with "data_batch" (training data)
train_fns = [fn for fn in listdir(dataset_path) if isfile(join(dataset_path, fn)) & fn.startswith("data_batch")]

# list sorting
train_fns.sort()

# make a glace about the training data
fn = train_fns[0]
raw_data = unpickle(dataset_path + fn)

# type of raw data
type(raw_data)
# <type 'dict'>

# check keys of training data
raw_data_keys = raw_data.keys()
# output ['data', 'labels', 'batch_label', 'filenames']

# check dimensions of ['data']
raw_data['data'].shape
# (10000, 3072)

# concatenate pixel (px) data into one ndarray [img_px_values]
# concatenate label data into one ndarray [img_lab]
img_px_values = 0
img_lab = 0
for fn in train_fns:
	raw_data = unpickle(dataset_path + fn)
	if fn == train_fns[0]:
		img_px_values = raw_data['data']
		img_lab = raw_data['labels']
	else:
		img_px_values = numpy.vstack((img_px_values, raw_data['data']))
		img_lab = numpy.hstack((img_lab, raw_data['labels']))

# convert 1d-ndarray (0:3072) to 3d-ndarray(32,32,3)
X_train = numpy.asarray([numpy.dstack((r[0:(width*height)].reshape(height,width),
									   r[(width*height):(2*width*height)].reshape(height,width),
									   r[(2*width*height):(3*width*height)].reshape(height,width)
									 )) for r in img_px_values])

Y_train = np_utils.to_categorical(numpy.array(img_lab), classes)

# draw one image from the pixel data
ndarray2image(X_train[0],"test_image.png")

# print the dimension of training data
print 'X_train shape:', X_train.shape
print 'Y_train shape:', Y_train.shape

''' read testing data '''
# get the file names which start with "test_batch" (testing data)
test_fns = [fn for fn in listdir(dataset_path) if isfile(join(dataset_path, fn)) & fn.startswith("test_batch")]

# read testing data
raw_data = unpickle(dataset_path + fn)

# type of raw data
type(raw_data)

# check keys of testing data
raw_data_keys = raw_data.keys()
# ['data', 'labels', 'batch_label', 'filenames']

img_px_values = raw_data['data']

# check dimensions of data
print "dim(data)", numpy.array(img_px_values).shape
# dim(data) (10000, 3072)

img_lab = raw_data['labels']
# check dimensions of labels
print "dim(labels)",numpy.array(img_lab).shape
# dim(data) (10000,)

X_test = numpy.asarray([numpy.dstack((r[0:(width*height)].reshape(height,width),
									  r[(width*height):(2*width*height)].reshape(height,width),
									  r[(2*width*height):(3*width*height)].reshape(height,width)
									)) for r in img_px_values])

Y_test = np_utils.to_categorical(numpy.array(raw_data['labels']), classes)

# scale image data to range [0, 1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

# print the dimension of training data
print 'X_test shape:', X_test.shape
print 'Y_test shape:', Y_test.shape

# normalize inputs from 0-255 to 0.0-1.0

'''CNN model'''
model = Sequential()
# 請建立一個 CNN model

# CNN

model.add(Flatten())

#DNN


'''setting optimizer'''
learning_rate = 0.01
learning_decay = 0.01/32
sgd = SGD(lr=learning_rate, decay=learning_decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# check parameters of every layers
model.summary()

''' training'''
batch_size = 128
epoch = 5
# validation data comes from training data
# model.fit(X_train, Y_train, batch_size=batch_size,
# 	      nb_epoch=epoch, validation_split=0.1, shuffle=True)

# validation data comes from testing data
fit_log = model.fit(X_train, Y_train, batch_size=batch_size,
	                nb_epoch=epoch, validation_data=(X_test, Y_test), shuffle=True)

'''saving training history'''
import csv
history_fn = 'cifar_10.csv'
with open(history_fn, 'wb') as csv_file:
	w = csv.writer(csv_file)
	temp = numpy.array(fit_log.history.values())
	w.writerow(fit_log.history.keys())
	for i in range(temp.shape[1]):
		w.writerow(temp[:,i])

'''saving model'''
from keras.models import load_model
model.save('cifar_10.h5')
del model

'''loading model'''
model = load_model('cifar_10.h5')

'''prediction'''
pred = model.predict_classes(X_test, batch_size, verbose=0)
ans = [numpy.argmax(r) for r in Y_test]

# caculate accuracy rate of testing data
acc_rate = sum(pred-ans == 0)/float(pred.shape[0])

print "Accuracy rate:", acc_rate