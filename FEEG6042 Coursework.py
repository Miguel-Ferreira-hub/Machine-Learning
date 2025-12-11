#FEEG6042: Coursework, effect of scaling on the training behaviour of a deep neural network
#Using CNN to classify image data from the mnist dataset
#Library imports
import numpy as np
import sklearn
import keras
import keras.datasets.mnist as mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz") #Extract dataset

#Display Data
for i in range(10):
    x = x_train[(y_train==i).squeeze(),:,:] 
    #Takes data from y classes, squeezes removes dimension size 1, eg (60000, 1) -> (60000,), 
    #: takes all data in the (28,28) dimensions
    plt.subplot(2,5,i+1) #Index to location
    plt.imshow(x[0], cmap='gray') #Takes the first image data to display, 28x28 image data
    plt.axis('off') #No plot axis
plt.tight_layout() #Ensures everything fits correctly
plt.show() #Show

#Control Experiment
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)) #Defines min-max scaler

def correct_scaler(x, scaler):
    #Per-image scaling of each pixel, darkest goes to 1, lightest goes to 0
    x_copy = x.copy().astype('float32') # Copy data so no modifications are done to the original
    for i, X in enumerate(x_copy[:,:,:]): #Index image data
        x_copy[i,:,:] = scaler.fit_transform(X.reshape(28*28,1)).reshape(28,28) #Applies min-max scaler to each image
    return x_copy #Returns scaled data

#Global Scaling
scaler12 = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)) #Scaler for 1.2, min-max scaler

def scaler12_func(x, scaler):
    #Flattens every pixel in the dataset to one long column and scales it
    x_copy = x.copy().astype('float32') #Stops modification to original data and converts to float
    x_copy = scaler.fit_transform(x_copy.reshape(-1,1)).reshape(x_copy.shape) #Scales globally
    return x_copy #returns scaled data

#Random scaling errors, correct scaler with random anomalies
scaler13 = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)) #Scaler for 1.3, min-max scaler

def random_image_scaling_errors(x, scaler, factor=100):
    x_copy = x.copy().astype('float32') #Copy to avoid modifying original data
    for i, X in enumerate(x_copy[:,:,:]): #iterates over each image, i is the index, X is a (28,28) image
        x_copy[i,:,:] = scaler.fit_transform(X.reshape(28*28,1)).reshape(28,28) #Flattens array to (784,1)to scale each pixel, scales images then reshapes back to (28,28)
    mask = np.random.rand(x_copy.shape[0]) < 0.50  #Around half of the pixels in the image get scaling errors
    x_copy[mask] *= factor #When mask is true applies error, should be around 50% of the time
    return x_copy #Returns scaled data

#Using standard scaler
scaler14 = sklearn.preprocessing.StandardScaler() #Defines standard scaler

def scaler14_func(x, scaler):
    #Per-image scaling of each pixel, this time with the standard
    x_copy = x.copy().astype('float32') # Copy data so no modifications are done to the original
    for i, X in enumerate(x_copy[:,:,:]): #Index image data
        x_copy[i,:,:] = scaler.fit_transform(X.reshape(28*28,1)).reshape(28,28) #Applies standard scaler to each image
    return x_copy #returns scaled data

X_train = correct_scaler(x_train, scaler) #Scaling control training data
X_test = correct_scaler(x_test, scaler) #Scaling control test data

X_test=X_test.reshape(-1,28,28,1) #Reshaping to correct format for neural network
X_train=X_train.reshape(-1,28,28,1) #Reshaping to correct format for neural network

#Output (y) scaling
onehot = sklearn.preprocessing.OneHotEncoder() #One-hot encoder, values either 0 or 1 for each class
onehot.fit(y_train.reshape(-1,1)) #Fits scaler
Y_train = onehot.transform(y_train.reshape(-1,1)).toarray() #Applies encoder and reshapes into format for cnn
Y_test = onehot.transform(y_test.reshape(-1,1)).toarray()

#Experiment 1, Incorrect Input Scaling
#No scaling
X_train11 = x_train.reshape(-1,28,28,1) #Reshapes into format for cnn
X_test11 = x_test.reshape(-1,28,28,1)

#Whole dataset scaled together
X_train12 = scaler12_func(x_train, scaler12) #Scaling for whole training dataset
X_test12 = scaler12_func(x_test, scaler12) #Scaling for whole test dataset

X_test12=X_test12.reshape(-1,28,28,1) #Reshape into format for cnn
X_train12=X_train12.reshape(-1,28,28,1) #Reshape into format for cnn

#Random errors in scaling
X_train13 = random_image_scaling_errors(x_train, scaler13) #Scale training data with random errors
X_test13 = random_image_scaling_errors(x_test, scaler13) #Scale test data with random errors

X_test13=X_test13.reshape(-1,28,28,1) #Reshape into format for cnn
X_train13=X_train13.reshape(-1,28,28,1) #Reshape into format for cnn

#Using standard scaler
X_train14 = scaler14_func(x_train, scaler14) #Standard scaler to each image in the training data
X_test14 = scaler14_func(x_test, scaler14) #Standard scaler to each image in the test data

X_test14=X_test14.reshape(-1,28,28,1) #Reshape into format for cnn
X_train14=X_train14.reshape(-1,28,28,1) #Reshape into format for cnn

#Output (y) scaling
onehot1 = sklearn.preprocessing.OneHotEncoder()
onehot1.fit(y_train.reshape(-1,1))
Y_train1 = onehot1.transform(y_train.reshape(-1,1)).toarray()
Y_test1 = onehot1.transform(y_test.reshape(-1,1)).toarray()

#Experiment 2, Incorrect Output Scaling
#No scaling for input or output features
X_train2 = X_train
X_test2 = X_test

Y_train21 = y_train
Y_test21 = y_test

#Using Standard Scaler
scaler22 = sklearn.preprocessing.StandardScaler() #Defining, fitting, applying and reshaping data using standard scaler
scaler22.fit(y_train.reshape(-1,1))
Y_train22 = scaler22.transform(y_train.reshape(-1,1))
Y_test22 = scaler22.transform(y_test.reshape(-1,1))

#Using MinMaxScaler
scaler23 = sklearn.preprocessing.MinMaxScaler() #Defining, fitting, applying and reshaping data using min-max scaler
scaler23.fit(y_train.reshape(-1,1))
Y_train23 = scaler23.transform(y_train.reshape(-1,1))
Y_test23 = scaler23.transform(y_test.reshape(-1,1))

def cnn_func(X_train, X_test, Y_train, Y_test):
    cnn = keras.models.Sequential()
    cnn.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=X_train.shape[1:])) 
    #Slides kernel (filter) over image, performs 1st convolution of values and returns a new image tensor, extracts meaningful features
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2))) #Slides kernel over the image and computes maximum value, keeps strongest features
    cnn.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu')) #Second convolution
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2))) #Second max-pooling layer
    cnn.add(keras.layers.Flatten()) #Flatten vector to output, 2D to 1D
    cnn.add(keras.layers.Dense(10, activation='softmax')) #Output layer, activation function for classification
    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = cnn.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=5, batch_size=64) #Train model
    return cnn, history

def cnn_func_reg(X_train, X_test, Y_train, Y_test):
    cnn = keras.models.Sequential() #CNN for regression
    cnn.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=X_train.shape[1:]))
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    cnn.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    cnn.add(keras.layers.Flatten())
    cnn.add(keras.layers.Dense(1)) #No activation function and 1 for output
    cnn.compile(loss='mse', optimizer='adam', metrics=['mse']) #using mse instead of categorical cross-entropy
    history = cnn.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=5, batch_size=64) 
    return cnn, history

def cnn_func_reg_control(X_train, X_test, Y_train, Y_test):
    cnn = keras.models.Sequential()
    cnn.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=X_train.shape[1:]))
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    cnn.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    cnn.add(keras.layers.Flatten())
    cnn.add(keras.layers.Dense(10))
    cnn.compile(loss='mse', optimizer='adam', metrics=['mse'])
    history = cnn.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=5, batch_size=64) 
    return cnn, history

def cnn_func_small(X_train, X_test, Y_train, Y_test): #Model 2
    cnn = keras.models.Sequential()
    cnn.add(keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=X_train.shape[1:])) 
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2))) 
    cnn.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu')) 
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2))) 
    cnn.add(keras.layers.Flatten()) 
    cnn.add(keras.layers.Dense(10, activation='softmax')) 
    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = cnn.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=5, batch_size=64) 
    return cnn, history

def cnn_func_large(X_train, X_test, Y_train, Y_test): #Model 3
    cnn = keras.models.Sequential()
    cnn.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=X_train.shape[1:])) 
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2))) 
    cnn.add(keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu')) 
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2))) 
    cnn.add(keras.layers.Flatten()) 
    cnn.add(keras.layers.Dense(10, activation='softmax'))
    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = cnn.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=5, batch_size=64) 
    return cnn, history

###################
#CONTORL EXPERIMENT
###################
cnn, history = cnn_func(X_train, X_test, Y_train, Y_test) #CNN function calls
cnn.summary() #Summary of layers and parameters
#######################################
#EXPERIMENT 1 - INCORRECT INPUT SCALING
#######################################
#No Scaling
cnn11, history11 = cnn_func(X_train11, X_test11, Y_train1, Y_test1)
#Global Scaling
cnn12, history12 = cnn_func(X_train12, X_test12, Y_train1, Y_test1)
#Random Errors
cnn13, history13 = cnn_func(X_train13, X_test13, Y_train1, Y_test1)
#Standard Scaler
cnn14, history14 = cnn_func(X_train14, X_test14, Y_train1, Y_test1)

########################################
#EXPERIMENT 2 - INCORRECT OUTPUT SCALING
########################################
#Control Experiment
cnn_control, history_control = cnn_func_reg_control(X_train2, X_test2, Y_train, Y_test)
#No Scaling
cnn21, history21 = cnn_func_reg(X_train2, X_test2, Y_train21, Y_test21)
#Min-Max Scaling
cnn22, history22 = cnn_func_reg(X_train2, X_test2, Y_train22, Y_test22)
#Standard Scaler
cnn23, history23 = cnn_func_reg(X_train2, X_test2, Y_train23, Y_test23)
cnn21.summary()

#Training behaviour experiment 1 - accuracy
plt.figure()
plt.semilogy(history.history['accuracy'], color='red') #Plot training accuracy
plt.semilogy(history.history['val_accuracy'], '--', color='red') #Plot test accuracy
plt.semilogy(history11.history['accuracy'], color='blue') 
plt.semilogy(history11.history['val_accuracy'], '--', color='blue')
plt.semilogy(history12.history['accuracy'], color='green') 
plt.semilogy(history12.history['val_accuracy'], '--', color='green')
plt.semilogy(history13.history['accuracy'], color='gold') 
plt.semilogy(history13.history['val_accuracy'], '--', color='gold')
plt.semilogy(history14.history['accuracy'], color='black') 
plt.semilogy(history14.history['val_accuracy'], '--', color='black')
plt.title('Experiment 1 Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Control Train', 'Control Test', 'No Scaling Train', 'No Scaling Test', 'Global Scaling Train', 'Global Scaling Test', 
'Random Errors Train', 'Random Errors Test', 'Standard Scaler Train', 'Standard Scaler Test'], loc='lower right')
plt.show() 
print(max(history.history['accuracy']))
print(max(history.history['val_accuracy']))
print(max(history11.history['accuracy']))
print(max(history11.history['val_accuracy']))
print(max(history12.history['accuracy']))
print(max(history12.history['val_accuracy']))
print(max(history13.history['accuracy']))
print(max(history13.history['val_accuracy']))
print(max(history14.history['accuracy']))
print(max(history14.history['val_accuracy']))

#Training behaviour experiment 1 with less results shown - accuracy
plt.figure()
plt.semilogy(history.history['accuracy'], color='red') 
plt.semilogy(history.history['val_accuracy'], '--', color='red')
plt.semilogy(history11.history['accuracy'], color='blue') 
plt.semilogy(history11.history['val_accuracy'], '--', color='blue')
plt.semilogy(history12.history['accuracy'], color='green') 
plt.semilogy(history12.history['val_accuracy'], '--', color='green')
plt.semilogy(history14.history['accuracy'], color='black') 
plt.semilogy(history14.history['val_accuracy'], '--', color='black')
plt.title('Experiment 1 Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Control Train', 'Control Test', 'No Scaling Train', 'No Scaling Test', 'Global Scaling Train', 
'Global Scaling Test', 'Standard Scaler Train', 'Standard Scaler Test'], loc='lower right')
plt.show() 

#Training behaviour experiment 1 - loss function
plt.figure()
plt.semilogy(history.history['loss'], color='red') 
plt.semilogy(history.history['val_loss'], '--', color='red')
plt.semilogy(history11.history['loss'], color='blue') 
plt.semilogy(history11.history['val_loss'], '--', color='blue')
plt.semilogy(history12.history['loss'], color='green') 
plt.semilogy(history12.history['val_loss'], '--', color='green')
plt.semilogy(history13.history['loss'], color='gold') 
plt.semilogy(history13.history['val_loss'], '--', color='gold')
plt.semilogy(history14.history['loss'], color='black') 
plt.semilogy(history14.history['val_loss'], '--', color='black')
plt.title('Experiment 1 Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Control Train', 'Control Test', 'No Scaling Train', 'No Scaling Test', 'Global Scaling Train', 'Global Scaling Test', 
'Random Errors Train', 'Random Errors Test', 'Standard Scaler Train', 'Standard Scaler Test'], loc='upper right')
plt.show() 

#RMSE calculation experiment 2
Y_control_pred = cnn_control.predict(X_test2)
y_control_pred = onehot.inverse_transform(Y_control_pred)
RMSE_control = np.sqrt(np.mean((y_control_pred-y_test)**2))

Y21_pred = cnn21.predict(X_test2)
y21_pred = Y21_pred
RMSE21 = np.sqrt(np.mean((y21_pred-y_test)**2))

Y22_pred = cnn22.predict(X_test2)
y22_pred = scaler22.inverse_transform(Y22_pred)
RMSE22 = np.sqrt(np.mean((y22_pred-y_test)**2))

Y23_pred = cnn23.predict(X_test2)
y23_pred = scaler23.inverse_transform(Y23_pred)
RMSE23 = np.sqrt(np.mean((y23_pred-y_test)**2))

#Print statements
print("RMSE for the control experiment: " + str(RMSE_control))
print("RMSE for no output scaling: " + str(RMSE21))
print("RMSE for scaling with standard scaler: " + str(RMSE22))
print("RMSE for scaling with min-max scaler: " + str(RMSE23))

#Experimnent 3 - Function calls
#Control Experiment
cnn_control_small, history_control_small = cnn_func_small(X_train, X_test, Y_train, Y_test)
cnn_control_large, history_control_large = cnn_func_large(X_train, X_test, Y_train, Y_test)
#No Scaling
cnn31, history31 = cnn_func_small(X_train11, X_test11, Y_train1, Y_test1)
cnn41, history41 = cnn_func_large(X_train11, X_test11, Y_train1, Y_test1)
#Global Scaling
cnn32, history32 = cnn_func_small(X_train12, X_test12, Y_train1, Y_test1)
cnn42, history42 = cnn_func_large(X_train12, X_test12, Y_train1, Y_test1)
#Random Errors
cnn33, history33 = cnn_func_small(X_train13, X_test13, Y_train1, Y_test1)
cnn43, history43 = cnn_func_large(X_train13, X_test13, Y_train1, Y_test1)
#Standard scaler
cnn34, history34 = cnn_func_small(X_train14, X_test14, Y_train1, Y_test1)
cnn44, history44 = cnn_func_large(X_train14, X_test14, Y_train1, Y_test1)

#Experiment 3 plots
plt.figure()
plt.semilogy(history.history['accuracy'], color='red') 
plt.semilogy(history.history['val_accuracy'], '--', color='red')
plt.semilogy(history_control_small.history['accuracy'], color='blue') 
plt.semilogy(history_control_small.history['val_accuracy'], '--', color='blue')
plt.semilogy(history_control_large.history['accuracy'], color='gold') 
plt.semilogy(history_control_large.history['val_accuracy'], '--', color='gold')
plt.title('Experiment 3: Control')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Model 1 Train', 'Model 1 Test', 'Model 2 Train', 'Model 2 Test', 'Model 3 Train', 'Model 3 Test'], loc='lower right')
plt.show() 

print(max(history.history['accuracy']))
print(max(history.history['val_accuracy']))
print(max(history_control_small.history['accuracy']))
print(max(history_control_small.history['val_accuracy']))
print(max(history_control_large.history['accuracy']))
print(max(history_control_large.history['val_accuracy']))

plt.figure()
plt.semilogy(history11.history['accuracy'], color='red') 
plt.semilogy(history11.history['val_accuracy'], '--', color='red')
plt.semilogy(history31.history['accuracy'], color='blue') 
plt.semilogy(history31.history['val_accuracy'], '--', color='blue')
plt.semilogy(history41.history['accuracy'], color='gold') 
plt.semilogy(history41.history['val_accuracy'], '--', color='gold')
plt.title('Experiment 3: No Scaling')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Model 1 Train', 'Model 1 Test', 'Model 2 Train', 'Model 2 Test', 'Model 3 Train', 'Model 3 Test'], loc='lower right')
plt.show() 

print(max(history11.history['accuracy']))
print(max(history11.history['val_accuracy']))
print(max(history31.history['accuracy']))
print(max(history31.history['val_accuracy']))
print(max(history41.history['accuracy']))
print(max(history41.history['val_accuracy']))

plt.figure()
plt.semilogy(history12.history['accuracy'], color='red') 
plt.semilogy(history12.history['val_accuracy'], '--', color='red')
plt.semilogy(history32.history['accuracy'], color='blue') 
plt.semilogy(history32.history['val_accuracy'], '--', color='blue')
plt.semilogy(history42.history['accuracy'], color='gold') 
plt.semilogy(history42.history['val_accuracy'], '--', color='gold')
plt.title('Experiment 3: Global Scaling')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Model 1 Train', 'Model 1 Test', 'Model 2 Train', 'Model 2 Test', 'Model 3 Train', 'Model 3 Test'], loc='lower right')
plt.show() 

print(max(history12.history['accuracy']))
print(max(history12.history['val_accuracy']))
print(max(history32.history['accuracy']))
print(max(history32.history['val_accuracy']))
print(max(history42.history['accuracy']))
print(max(history42.history['val_accuracy']))

plt.figure()
plt.semilogy(history13.history['accuracy'], color='red') 
plt.semilogy(history13.history['val_accuracy'], '--', color='red')
plt.semilogy(history33.history['accuracy'], color='blue') 
plt.semilogy(history33.history['val_accuracy'], '--', color='blue')
plt.semilogy(history43.history['accuracy'], color='gold') 
plt.semilogy(history43.history['val_accuracy'], '--', color='gold')
plt.title('Experiment 3: Scaling with Random Errors')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Model 1 Train', 'Model 1 Test', 'Model 2 Train', 'Model 2 Test', 'Model 3 Train', 'Model 3 Test'], loc='lower right')
plt.show() 

print(max(history13.history['accuracy']))
print(max(history13.history['val_accuracy']))
print(max(history33.history['accuracy']))
print(max(history33.history['val_accuracy']))
print(max(history43.history['accuracy']))
print(max(history43.history['val_accuracy']))

plt.figure()
plt.semilogy(history14.history['accuracy'], color='red') 
plt.semilogy(history14.history['val_accuracy'], '--', color='red')
plt.semilogy(history34.history['accuracy'], color='blue') 
plt.semilogy(history34.history['val_accuracy'], '--', color='blue')
plt.semilogy(history44.history['accuracy'], color='gold') 
plt.semilogy(history44.history['val_accuracy'], '--', color='gold')
plt.title('Experiment 3: Standard Scaler')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Model 1 Train', 'Model 1 Test', 'Model 2 Train', 'Model 2 Test', 'Model 3 Train', 'Model 3 Test'], loc='lower right')
plt.show() 

print(max(history14.history['accuracy']))
print(max(history14.history['val_accuracy']))
print(max(history34.history['accuracy']))
print(max(history34.history['val_accuracy']))
print(max(history44.history['accuracy']))
print(max(history44.history['val_accuracy']))

cnn31.summary()