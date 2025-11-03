#FEEG6042: Introduction to Machine Learning Lab 3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
from keras.models import clone_model
import keras
import sklearn.preprocessing
#import keras.datasets.cifar10 as cifar

#Load Data
(x_tr, y_tr), (x_te, y_te)=keras.datasets.cifar10.load_data()
#(x_tr, y_tr), (x_te, y_te)=cifar.load_data()

#Plotting data
for i in range(10):
    x = x_tr[(y_tr==i).squeeze(),:,:,:] #Takes data from y classes, squeezes removes dimension size 1, eg (50000, 1) -> (50000,), : takes all data in dimensions
    plt.subplot(2,5,i+1) #Index to location
    plt.imshow(x[0]) #Takes the first image data to display, 32x32 pixels with RGB color channels
    plt.axis('off') #No plot axis
plt.show() #Show

#print(y_tr, y_te) #Integer Label Format
#Cats y_tr==3, first 10 images
for i in range(10):
    x = x_tr[(y_tr==3).squeeze(),:,:,:]
    plt.subplot(2,5,i+1)
    plt.imshow(x[i])
    plt.axis('off')
plt.show()

#Dogs y_tr==5, first 10 images
for i in range(10):
    x = x_tr[(y_tr==5).squeeze(),:,:,:]
    plt.subplot(2,5,i+1)
    plt.imshow(x[i])
    plt.axis('off')
plt.show()

#Filter data for cats (y_tr==3) and dogs (y_tr==5) only
cat_dog_train = (y_tr.squeeze() == 3) | (y_tr.squeeze() == 5)
x_train = x_tr[cat_dog_train].astype('float32')
y_train = y_tr[cat_dog_train].astype('float32')

cat_dog_test = (y_te.squeeze() == 3) | (y_te.squeeze() == 5)
x_test = x_te[cat_dog_test].astype('float32')
y_test = y_te[cat_dog_test].astype('float32')

#print(x_train.shape, x_test.shape)

#Image dimensions in (:,:,:,:) first index is the image, then 2D pixels, then RGB - made of 3 images
plt.subplot(1,2,1)
plt.imshow(x_train[399,:,:,:]/255)
plt.subplot(1,2,2)
plt.imshow(x_train[6199,:,:,:]/255)
plt.axis('off')
plt.show()

#Scaling pixels between 0,1
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))

def im_scaler(x,scaler):
    for i, X in enumerate(x[:,:,:,:]):
        x[i,:,:,:] = scaler.fit_transform(X.reshape(32*32*3,1)).reshape(32,32,3)
    return x

X_train = im_scaler(x_train,scaler)
X_test = im_scaler(x_test,scaler)

#Encoding
onehot = sklearn.preprocessing.OneHotEncoder()
onehot.fit(y_train.reshape(-1,1))
Y_train = onehot.transform(y_train.reshape(-1,1)).toarray()
Y_test = onehot.transform(y_test.reshape(-1,1)).toarray()

#Fully Connected Neural Network Model - Deep Neural Network
FCNN = keras.models.Sequential()
FCNN.add(keras.layers.Flatten(input_shape=X_train.shape[1:]))
FCNN.add(keras.layers.Dense(50, activation='relu'))
FCNN.add(keras.layers.Dense(25, activation='relu'))
FCNN.add(keras.layers.Dense(12, activation='relu'))
FCNN.add(keras.layers.Dense(5, activation='relu'))
FCNN.add(keras.layers.Dense(3, activation='relu'))
FCNN.add(keras.layers.Dense(2, activation='softmax')) #softmax activation function is required in the last layer for categorical cross-entropy method
FCNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = FCNN.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=20, batch_size=64) #Train model

#Results of optimisation
plt.semilogy(history.history['accuracy']) #Plot how the model performed during training
plt.semilogy(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.semilogy(history.history['loss']) #Plot how the model performed during training
plt.semilogy(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Convolution Neural Network Model
cnn = keras.models.Sequential()
cnn.add(keras.layers.Conv2D(8, kernel_size=(3,3), padding='same', input_shape=X_train.shape[1:], activation='relu'))
cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
cnn.add(keras.layers.Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))
cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
cnn.add(keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
cnn.add(keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
cnn.add(keras.layers.Flatten())
cnn.add(keras.layers.Dense(2, activation='softmax'))
cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
historycnn = cnn.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=64)
cnn.summary()

#Results of optimisation
plt.semilogy(historycnn.history['accuracy']) #Plot how the model performed during training
plt.semilogy(historycnn.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.semilogy(historycnn.history['loss']) #Plot how the model performed during training
plt.semilogy(historycnn.history['val_loss'])
plt.title('Loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print(cnn.count_params())
print(FCNN.count_params())