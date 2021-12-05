import pandas as pd
import sklearn as sl
from IPython import display
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import sklearn as sl
from IPython import display
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from keras.datasets import mnist
from keras.utils import np_utils
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# path = 'F:\\git\\python\\test-case-generator\\src\\test\\example_4\\data\\mnist.npz'
path = 'data/mnist.npz'
f = np.load(path)
train_images,train_labels = f['x_train'],f['y_train']
test_images ,test_labels = f['x_test'],f['y_test']
f.close()
X_train = X_train.reshape([60000,784])
X_test = X_test.reshape([10000,784])
print(X_train.shape,X_test.shape)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_train = X_train/255
X_test = X_test/255
train_labels = to_categorical(y_train)
test_labels = to_categorical(y_test)
# 多层感知机模型
model = Sequential()
model.add(Dense(10,input_shape=(28*28,)))
model.add(Activation('softmax'))
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(X_train, train_labels, epochs=200, batch_size=128, verbose=True, validation_split=0.2)
# 评估神经网络
test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=True)
print('test_loss', test_loss, '\n', 'test_acc', test_acc)

# 修改模型
model = Sequential()
model.add(Dense(128, input_shape=(28*28, ),activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(X_train, train_labels, epochs=200, batch_size=128, verbose=True, validation_split=0.2)
test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=True)
print('test_loss', test_loss, '\n', 'test_acc', test_acc)

# 接下来使用cnn做实验
