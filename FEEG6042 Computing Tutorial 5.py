#FEEG6042: Computing Tutorial 5 - Recurrent Neural Network to classify sound
import numpy as np
import sklearn
import keras
import matplotlib.pyplot as plt

#Load Data
#Mel Frequency Cepstral Coefficients
x= np.load("X_mfcc_10000.npy")
y= np.load("Y_mfcc_10000.npy")

#Pre-processing
x = np.moveaxis(x, [0, 1, 2], [0, 2, 1]) #Swaps temporal and frequency features, this is the form keras RNNs take as input
print(x.shape)
#Scaling Method 1 - Datapoint wise scaling
def Method1(x):
    x_scaled = x.copy()
    for i in range(x.shape[0]):
        scaler = sklearn.preprocessing.StandardScaler()
        x_scaled[i] = scaler.fit_transform(x[i])
    return x_scaled

X1 = Method1(x)

X_train1, X_test1, Y_train1, Y_test1 = sklearn.model_selection.train_test_split(X1, y, test_size=0.1)

plt.figure
plt.imshow(X1[0])
plt.colorbar()
plt.show()

print('Printing X1:...')
print(X1)
#Scaling Method 2 - Datapoint wise scaling and multiplied by 100
def Method2(x):
    x_scaled = x.copy()
    for i in range(x.shape[0]):
        scaler = sklearn.preprocessing.StandardScaler()
        x_scaled[i] = scaler.fit_transform(x[i]) * 100
    return x_scaled

X2 = Method2(x)

X_train2, X_test2, Y_train2, Y_test2 = sklearn.model_selection.train_test_split(X2, y, test_size=0.1)

plt.figure
plt.imshow(X2[0])
plt.colorbar()
plt.show()

print('Printing X2:...')
print(X2)
#Scaling Method 3 - MinMaxScaler between 0 and 1
def Method3(x):
    x = x.copy()
    N = x.shape[0]

    for i in range(N):
        scaler = sklearn.preprocessing.MinMaxScaler()
        x[i] = scaler.fit_transform(x[i])
    return x

X3 = Method3(x)

X_train3, X_test3, Y_train3, Y_test3 = sklearn.model_selection.train_test_split(X3, y, test_size=0.1)

plt.figure
plt.imshow(X3[0])
plt.colorbar()
plt.show()

print('Printing X3:...')
print(X3)
#Scaling Method 4 - Feature wise scaling
X4=x/np.std(x,axis=0)

X_train4, X_test4, Y_train4, Y_test4 = sklearn.model_selection.train_test_split(X4, y, test_size=0.1)

plt.figure
plt.imshow(X4[0])
plt.colorbar()
plt.show()

print('Printing X4:...')
print(X4)

#Models
model1 = keras.Sequential()
model1.add(keras.layers.LSTM(16))
model1.add(keras.layers.Dense(2, activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history1 = model1.fit(X_train1, Y_train1, validation_data = (X_test1,Y_test1), epochs=20, batch_size=64) #Train model
model1.summary()

model2 = keras.Sequential()
model2.add(keras.layers.LSTM(16))
model2.add(keras.layers.Dense(2, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history2 = model2.fit(X_train2, Y_train2, validation_data = (X_test2,Y_test2), epochs=20, batch_size=64) #Train model
model2.summary()

model3 = keras.Sequential()
model3.add(keras.layers.LSTM(16))
model3.add(keras.layers.Dense(2, activation='softmax'))
model3.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history3 = model3.fit(X_train3, Y_train3, validation_data = (X_test3,Y_test3), epochs=20, batch_size=64) #Train model
model3.summary()

model4 = keras.Sequential()
model4.add(keras.layers.LSTM(16))
model4.add(keras.layers.Dense(2, activation='softmax'))
model4.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history4 = model4.fit(X_train4, Y_train4, validation_data = (X_test4,Y_test4), epochs=20, batch_size=64) #Train model
model4.summary()

#Loss
plt.semilogy(history1.history['loss']) 
plt.semilogy(history1.history['val_loss'], '--')
plt.semilogy(history2.history['loss']) 
plt.semilogy(history2.history['val_loss'], '--')
plt.semilogy(history3.history['loss']) 
plt.semilogy(history3.history['val_loss'], '--')
plt.semilogy(history4.history['loss']) 
plt.semilogy(history4.history['val_loss'], '--')
plt.title('Loss for the 4 models')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Model 1 Train', 'Model 1 Test', 'Model 2 Train', 'Model 2 Test', 'Model 3 Train', 'Model 3 Test', 'Model 4 Train', 'Model 4 Test'], loc='upper left')
plt.show()

#Accuracy
plt.semilogy(history1.history['accuracy']) 
plt.semilogy(history1.history['val_accuracy'], '--')
plt.semilogy(history2.history['accuracy']) 
plt.semilogy(history2.history['val_accuracy'], '--')
plt.semilogy(history3.history['accuracy']) 
plt.semilogy(history3.history['val_accuracy'], '--')
plt.semilogy(history4.history['accuracy']) 
plt.semilogy(history4.history['val_accuracy'], '--')
plt.title('Accuracy for the 4 models')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Model 1 Train', 'Model 1 Test', 'Model 2 Train', 'Model 2 Test', 'Model 3 Train', 'Model 3 Test', 'Model 4 Train', 'Model 4 Test'], loc='upper left')
plt.show()