# libraries & packages
import numpy
import math
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

# set dataset path
dataset_path = '../cifar_10/'
exec(open("read_dataset2img.py").read())

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
	                epochs=epoch, validation_data=(X_test, Y_test), shuffle=True)

'''saving training history'''
import csv
exec(open("write_csv.py").read())

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

print ("Accuracy rate:", acc_rate)