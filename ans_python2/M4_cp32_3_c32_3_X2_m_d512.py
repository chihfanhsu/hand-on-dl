# libraries & packages
import numpy
import math
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

# set dataset path
dataset_path = '../cifar_10/'
execfile('read_dataset2img.py')

'''CNN model'''
model = Sequential()
model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=X_train[0].shape))
model.add(Activation('relu'))
model.add(Conv2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(classes))
model.add(Activation('softmax'))

'''setting optimizer'''
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
model.compile(loss= 'categorical_crossentropy',
              		optimizer='Adam',
              		metrics=['accuracy'])


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
output_fn = 'M4_cp32_3_c32_3_X2_m_d512'
execfile('write_csv.py')

'''saving model'''
from keras.models import load_model
model.save(output_fn + '.h5')
del model

'''loading model'''
model = load_model(output_fn + '.h5')

'''prediction'''
pred = model.predict_classes(X_test, batch_size, verbose=0)
ans = [numpy.argmax(r) for r in Y_test]

# caculate accuracy rate of testing data
acc_rate = sum(pred-ans == 0)/float(pred.shape[0])

print "Accuracy rate:", acc_rate
