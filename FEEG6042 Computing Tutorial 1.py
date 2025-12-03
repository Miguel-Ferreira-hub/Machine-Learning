#FEEG6042: Computing Tutorial 1
from keras.src.backend.jax import cudnn_ok
import numpy as np
import sklearn
import keras
import matplotlib.pyplot as plt
import keras.datasets.mnist as mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
#Plotting data
for i in range(10):
    x = x_train[(y_train==i).squeeze(),:,:] #Takes data from y classes, squeezes removes dimension size 1, eg (50000, 1) -> (50000,), : takes all data in dimensions
    plt.subplot(2,5,i+1) #Index to location
    plt.imshow(x[0]) #Takes the first image data to display, 32x32 pixels with RGB color channels
    plt.axis('off') #No plot axis
plt.show() #Show

#print(y_train, y_train) #Integer Label Format

#Scaler
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))

def im_scaler(x,scaler):
    for i, X in enumerate(x[:,:,:]):
        x[i,:,:] = scaler.fit_transform(X.reshape(28*28,1)).reshape(28,28)
    return x

X_train = im_scaler(x_train,scaler)
X_test = im_scaler(x_test,scaler)

X_test=X_test.reshape(-1,28,28,1)
X_train=X_train.reshape(-1,28,28,1)

#Encoding
onehot = sklearn.preprocessing.OneHotEncoder()
onehot.fit(y_train.reshape(-1,1))
Y_train = onehot.transform(y_train.reshape(-1,1)).toarray()
Y_test = onehot.transform(y_test.reshape(-1,1)).toarray()

#CNN
cnn = keras.models.Sequential()
cnn.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu',
input_shape=X_train.shape[1:]))
cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
cnn.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
cnn.add(keras.layers.Flatten())
cnn.add(keras.layers.Dense(10, activation='softmax'))
cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = cudnn_ok.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=20, batch_size=64) #Train model

#Results of optimisation
plt.semilogy(history.history['accuracy']) #Plot how the model performed during training
plt.semilogy(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()