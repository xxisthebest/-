import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
(train_data, train_label), (test_data, test_label) = cifar10.load_data()
x_data = train_data.astype('float32') / 255.
y_data = test_data.astype('float32') / 255.
import numpy as np
def one_hot(label, num_classes):
    label_one_hot = np.eye(num_classes)[label]
    return label_one_hot
num_classes = 10
train_label = train_label.astype('int32')
train_label = np.squeeze(train_label)
x_label = one_hot(train_label, num_classes)
test_label = test_label.astype('int32')
y_label = np.squeeze(test_label)
from keras import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
cnn = Sequential()
#unit1
cnn.add(Convolution2D(32, kernel_size=[3, 3], input_shape=(32, 32, 3), activation='relu', padding='same'))
cnn.add(Convolution2D(32, kernel_size=[3, 3], activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=[2, 2], padding='same'))
cnn.add(Dropout(0.5))
#unit2
cnn.add(Convolution2D(64, kernel_size=[3, 3], activation='relu', padding='same'))
cnn.add(Convolution2D(64, kernel_size=[3, 3], activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=[2, 2], padding='same'))
cnn.add(Dropout(0.5))
cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))

cnn.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
loss='categorical_crossentropy', metrics=['acc'])
history_cnn = cnn.fit(x_data, x_label, epochs=50, batch_size=32, shuffle=True, verbose=1, validation_split=0.1)
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(np.array(history_cnn.history['loss']))
plt.plot(np.array(history_cnn.history['val_loss']))
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.legend(['loss', 'val_loss'])
plt.show()
plt.figure(2)
plt.plot(np.array(history_cnn.history['acc']))
plt.plot(np.array(history_cnn.history['val_acc']))
plt.xlabel('Epoch')
plt.ylabel('Train acc')
plt.legend(['acc', 'val_acc'])
plt.show()
