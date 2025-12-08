#FEEG6042: Computing Tutorial 2, convolutional neural network that is also an auto-encoder
#from keras.src.backend.jax import cudnn_ok
import numpy as np
import sklearn
import keras
import matplotlib.pyplot as plt
import keras.datasets.mnist as mnist
from keras.models import clone_model

Standard_Scaler = False

(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
#Plotting data
for i in range(10):
    x = x_train[(y_train==i).squeeze(),:,:] #Takes data from y classes, squeezes removes dimension size 1, eg (50000, 1) -> (50000,), : takes all data in dimensions
    plt.subplot(2,5,i+1) #Index to location
    plt.imshow(x[0]) #Takes the first image data to display, 32x32 pixels with RGB color channels
    plt.axis('off') #No plot axis
plt.show() #Show

#Scaler
if Standard_Scaler == True:
    scaler = sklearn.preprocessing.StandardScaler()
else:
    scaler = sklearn.preprocessing.MinMaxScaler()

def im_scaler(x,scaler):
    for i, X in enumerate(x[:,:,:]):
        x[i,:,:] = scaler.fit_transform(X.reshape(28*28,1)).reshape(28,28)
    return x

X_train = im_scaler(x_train,scaler)
X_test = im_scaler(x_test,scaler)

X_test=X_test.reshape(-1,28,28,1)
X_train=X_train.reshape(-1,28,28,1)

Y_train = X_train
Y_test = X_test

for i in range(10):
    x = X_train[(y_train==i).squeeze(),:,:] #Takes data from y classes, squeezes removes dimension size 1, eg (50000, 1) -> (50000,), : takes all data in dimensions
    plt.subplot(2,5,i+1) #Index to location
    plt.imshow(x[0]) #Takes the first image data to display, 32x32 pixels with RGB color channels
    plt.axis('off') #No plot axis
plt.show() #Show

#Convolutional Auto-encoder
c = 30
cnn = keras.models.Sequential()
cnn.add(keras.layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), activation='relu', input_shape=X_train.shape[1:]))
cnn.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu'))
cnn.add(keras.layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), activation='relu'))
cnn.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
cnn.add(keras.layers.Flatten())
cnn.add(keras.layers.Dense(100, activation='sigmoid'))
cnn.add(keras.layers.Dense(c, activation='sigmoid'))
cnn.add(keras.layers.Dense(7*7*32, activation='sigmoid'))
cnn.add(keras.layers.Reshape((7,7,32)))
cnn.add(keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2)))
cnn.add(keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2)))
cnn.add(keras.layers.Conv2DTranspose(1, (3, 3), activation='relu', padding='same' ))
cnn.compile(loss='MSE', optimizer='adam', metrics=['MSE']) #Auto-encoder is a regression algorithm so here we use the MSE
history = cnn.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=5, batch_size=64) #Train model

#Results of optimisation
plt.semilogy(history.history['MSE']) #Plot how the model performed during training
plt.semilogy(history.history['val_MSE'])
plt.title('MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Extracting Sub-Models in Keras
def extract_layers(main_model, starting_layer_ix, ending_layer_ix): #defines a helper that copies a slice of layers from the Keras model
    # create an empty model
    new_model = keras.Sequential()
    for ix in range(starting_layer_ix, ending_layer_ix + 1): #Iterate over desired layers
        curr_layer = main_model.get_layer(index=ix) #grab each leayer from the model
        # copy this layer over to the new model
        new_model.add(curr_layer) #Add layer to this model
    return new_model

encoder = extract_layers(cnn,0,6) #Build an encoder sub-model from layers 0-6
decoder = extract_layers(cnn,7,11) #Build a decoder sub-model from layers 7-11
code1 = encoder.predict(X_test[0].reshape(-1,28,28,1)) #Encode first test sample and reshape
decode1 = decoder.predict(code1) #Feed into the decoder to reconstruct sample
code2 = encoder.predict(X_test[1].reshape(-1,28,28,1)) #Repeat encoding for the second sample
decode2 = decoder.predict(code2) #Feed into decoder to reconstruct sample

#Using code to interpolate - make sure to use min-max scaler
for i,c in enumerate(np.arange(0,1,0.1)):
    print(c)
    code = c*code1+(1-c)*code2
    decode = decoder.predict(code)
    plt.subplot(2,5,i+1)
    plt.imshow(decode.reshape((28,28)),cmap='gray')
    plt.axis('off')
    plt.tight_layout()

#Direct interpolation of images
for i,c in enumerate(np.arange(0,1,0.1)):
    print(c)
    interp = c*X_test[1].reshape(28,28)+(1-c)*X_test[0].reshape(28,28)
    plt.subplot(2,5,i+1)
    plt.imshow(interp,cmap='gray')
    plt.axis('off')
    plt.tight_layout()

#Using the auto-encoder to de-noise images
#Images with Gaussian noise
X_trainN = np.random.poisson(0.05+X_train)
X_testN = np.random.poisson(0.05+X_test)
for i in np.arange(5):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train[i].reshape(28,28),cmap='gray')
    plt.axis('off')
    plt.subplot(2,5,i+1+5)
    plt.imshow(X_trainN[i].reshape(28,28),cmap='gray')
    plt.axis('off')
    plt.tight_layout()

#Clone model here used to copy the model not assign a different name to the same model
daa = clone_model(cnn)
daa.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = daa.fit(X_trainN, Y_train, validation_data = (X_testN,Y_test), epochs=5, batch_size=64)

