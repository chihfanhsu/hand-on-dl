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
execfile('read_dataset2vec.py')

'''CNN model'''
model = Sequential()
model.add(Dense(512, input_shape=X_train[0].shape))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(classes))
model.add(Activation('softmax'))

'''setting optimizer'''
learning_rate = 0.01
learning_decay = 0.01/32
sgd = SGD(lr=learning_rate, decay=learning_decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# check parameters of every layers
model.summary()

''' training'''
batch_size = 128
epoch = 32
# validation data comes from training data
# model.fit(X_train, Y_train, batch_size=batch_size,
# 	      nb_epoch=epoch, validation_split=0.1, shuffle=True)

# validation data comes from testing data
fit_log = model.fit(X_train, Y_train, batch_size=batch_size,
	                nb_epoch=epoch, validation_data=(X_test, Y_test), shuffle=True)


'''saving training history'''
output_fn = 'M0_d512_512_512'
execfile('write_csv.py')


'''saving model'''
# from keras.models import load_model
# model.save(output_fn + '.h5')
# del model

'''loading model'''
# model = load_model(output_fn + '.h5')

'''prediction'''
pred = model.predict_classes(X_test, batch_size, verbose=0)
ans = [numpy.argmax(r) for r in Y_test]

# caculate accuracy rate of testing data
acc_rate = sum(pred-ans == 0)/float(pred.shape[0])

print "Accuracy rate:", acc_rate

