#FEEG6042: Computing Tutorial 3
import numpy as np
import sklearn
import keras
import matplotlib.pyplot as plt

#Load Data
#Mel Frequency Cepstral Coefficients
x= np.load("X_mfcc_10000.npy")
y= np.load("Y_mfcc_10000.npy")

#Pre-processing
x = np.moveaxis(x, [0, 1, 2], [0, 2, 1]) #Swaps temporal and frequency features, this is the form keras RNNs take

#Scaling - Data can be negative, unit variance but not zero mean
def MyScaler(X):
    for i, x in enumerate(X):
        X[i,:,:]=X[i,:,:]/np.std(x)
    return X

X=MyScaler(x)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1)

#Visualise Data
plt.figure
plt.imshow(x[0])
plt.show()

#Model Architecture, in this model only the last output from the LSTM layer is used so that data with different dimensionality can be used.
#LSTM generates 126 outputs
model1 = keras.Sequential()
model1.add(keras.layers.LSTM(16))
model1.add(keras.layers.Dense(2, activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history1 = model1.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=20, batch_size=64) #Train model
model1.summary()

#Results of optimisation
plt.semilogy(history1.history['accuracy']) #Plot how the model performed during training
plt.semilogy(history1.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Model2 returning all outputs from the LSTM layer
model2 = keras.Sequential()
model2.add(keras.layers.LSTM(16, return_sequences=True))
model2.add(keras.layers.Flatten())
model2.add(keras.layers.Dense(2,activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history2 = model2.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=20, batch_size=64) 
model2.summary()

#Results of optimisation
plt.semilogy(history2.history['accuracy']) 
plt.semilogy(history2.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Model3 normal NN
model3 = keras.Sequential()
model3.add(keras.layers.Flatten(input_shape=(126, 40)))
model3.add(keras.layers.Dense(28, activation='relu'))
model3.add(keras.layers.Dense(8, activation='relu'))
model3.add(keras.layers.Dense(2, activation='softmax'))
model3.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history3 = model3.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=20, batch_size=64) 
model3.summary()

#Results of optimisation
plt.semilogy(history3.history['accuracy']) 
plt.semilogy(history3.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Model4 - Adapted LSTM network
model4 = keras.Sequential()
model4.add(keras.layers.LSTM(26))
model4.add(keras.layers.Dense(16, activation='relu'))
model4.add(keras.layers.Dense(2, activation='softmax'))
model4.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history4 = model4.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=20, batch_size=64) #Train model
model4.summary()

#Results of optimisation
plt.semilogy(history1.history['accuracy']) 
plt.semilogy(history1.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Plot of the 4 models - Loss
plt.semilogy(history1.history['loss']) 
plt.semilogy(history1.history['val_loss'], '--')
plt.semilogy(history2.history['loss']) 
plt.semilogy(history2.history['val_loss'], '--')
plt.semilogy(history3.history['loss']) 
plt.semilogy(history3.history['val_loss'], '--')
plt.semilogy(history4.history['loss']) 
plt.semilogy(history4.history['val_loss'], '--')
plt.title('Loss values across the 4 models')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Plot of the 4 models - Accuracy
plt.semilogy(history1.history['accuracy']) 
plt.semilogy(history1.history['val_accuracy'], '--')
plt.semilogy(history2.history['accuracy']) 
plt.semilogy(history2.history['val_accuracy'], '--')
plt.semilogy(history3.history['accuracy']) 
plt.semilogy(history3.history['val_accuracy'], '--')
plt.semilogy(history4.history['accuracy']) 
plt.semilogy(history4.history['val_accuracy'], '--')
plt.title('Accuracy values across the 4 models')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()