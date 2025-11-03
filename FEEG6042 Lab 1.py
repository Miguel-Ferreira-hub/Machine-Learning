#FEEG6042 - Introduction to Machine Learning Lab 1
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import sklearn
import os

data=np.zeros((0,11)) #Initialize data matrix
directory = r'C:\Users\migue\Desktop\SpotifyData' #Directory of the data
for year in range(1950,2020,10): #Loop through the years
    file_path = os.path.join(directory, f'{year}.csv') #Path to the data
    df = pd.read_csv(file_path) #Read the data
    tmp = df.to_numpy()[:,4:].astype('double') #Double precision numbers 64-bit, less rounding erros/more precise, Single precision is 32-bit
    #data = df.to_numpy()[0,:4] #Extract the first row of the data up to column 4
    data = np.concatenate((data,tmp), axis=0) #Concatenate the data adds new dataset to the existing data (at the bottom of the matrix)

column_names = df.columns.tolist()

x = data[:,:10] #Extract the first 10 columns of the data
y = data[:,10:11] #Extract the last column of the data

#HISTOGRAM PLOTS
fig, axes = plt.subplots(2, 5, figsize=(15, 8)) #Defines axes and figure, 2 rows, 5 columns, size 15x8
fig.suptitle('Histograms of All Feature Distributions', fontsize=16) #Title of the figure with font size 16

for i in range(10):
    row = i // 5 #Row index - Calculates where to allocate data for each row and column of the subplot
    col = i % 5 #Column index
    
    # Calculate bins for each feature
    min_value = data[:, i].min() #Minimum value of the feature
    max_value = data[:, i].max() #Maximum value of the feature
    bins = np.linspace(min_value, max_value, 30) #Bins for the histogram
    
    axes[row, col].hist(data[:, i], bins=bins, alpha=0.7, edgecolor='black') #Plot the histogram
    axes[row, col].set_title(f'{column_names[i+4]}') #Title of the histogram
    axes[row, col].set_xlabel(f'{column_names[i+4]}') #Label of the x-axis
    axes[row, col].set_ylabel('Frequency') #Label of the y-axis
    axes[row, col].grid(True, alpha=0.3) #Grid of the histogram, transparent 30%

plt.tight_layout() #Adjust layout to prevent overlap
plt.show() #Show the plot

#SCATTER PLOTS
fig, axes = plt.subplots(2, 5, figsize=(15,8))
fig.suptitle('Scatter plots of output against all features', fontsize=16)

for i in range(10):
    row = i // 5 
    col = i % 5 

    axes[row, col].plot(x[:,i],y[:,0], 'o', markersize=3)
    axes[row, col].set_title(f'{column_names[14]} against {column_names[i+4]}') 
    axes[row, col].set_xlabel(f'{column_names[i+4]}') 
    axes[row, col].set_ylabel(f'{column_names[14]}') 
    axes[row, col].grid(True, alpha=0.3) 

plt.tight_layout()
plt.show()

#MACHINE LEARNING MODEL
#I Simple Linear Regression
#II Artificial Neural Network
#Evaluation Criterion - Mean Squared Error

#Splitting Test and Train Data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)
#print(x_test, y_test)

#Scaling Data
scalerX = sklearn.preprocessing.StandardScaler() #Imported Scalers for X
scalerY = sklearn.preprocessing.StandardScaler() #Imported Scalers for Y
scalerX.fit(x_train) #Fitting x training data
scalerY.fit(y_train) #Fitting y training data
#We train the scaling model on the training data but apply it later on the training and test data. Upper case X and Y denote scaled data
X_train = scalerX.transform(x_train)
X_test = scalerX.transform(x_test) 
Y_train = scalerY.transform(y_train)
Y_test = scalerY.transform(y_test)

#Mean and Standard Deviation of Training and Test Data (Scaled Data)
X_train_mean = np.mean(X_train, axis=0) #axis 0 represents columns and axis 1 represents rows
X_test_mean = np.mean(X_test, axis=0)
Y_train_mean = np.mean(Y_train, axis=0)
Y_test_mean = np.mean(Y_test, axis=0)
X_train_std = np.std(X_train, axis=0) 
X_test_std = np.std(X_test, axis=0)
Y_train_std = np.std(Y_train, axis=0)
Y_test_std = np.std(Y_test, axis=0)

print(f'Xtrain mean: {X_train_mean}') #Print Statements
print(f'Ytrain mean: {Y_train_mean}')
print(f'Xtrain std: {X_train_std}')
print(f'Ytrain std: {Y_train_std}')
print(f'Xtest mean: {X_test_mean}')
print(f'Ytest mean: {Y_test_mean}')
print(f'Xtest std: {X_test_std}')
print(f'Ytest std: {Y_test_std}')

#Training Linear Regression Model
model = sklearn.linear_model.LinearRegression() #Import Linear Regression Model
model.fit(X_train,Y_train[:,0:1]) #Fit Model to Train Data

Y_pred = model.predict(X_test) #Predict from Test Data

#Mean Squared Error
mse = np.mean((Y_pred-Y_test)**2) #MSE Equation
print("MSE on test data: " + str(mse)) #Print Statement

#Relative Mean Squared Error
relmse = np.mean((Y_pred-Y_test)**2)/ np.mean((Y_test)**2) #Equation
print("Relative MSE on test data: " + str(relmse)) #Print Statement

#Transforming Prediction Back to Original Scale and Working out RMSE
y_pred = scalerY.inverse_transform(Y_pred) #Inverse Transform
rmse = np.sqrt(np.mean((y_pred-y_test)**2)) #RMSE Equation
print("RMSE in popularity: " + str(rmse)) #Print Statement

#Histogram of Error
min_value_pred=y_pred[:,0].min()
max_value_pred=y_pred[:,0].max()
min_value_actual=y[:,0].min()
max_value_actual=y[:,0].max()
plt.hist(y_pred[:,0], bins=np.arange(min_value_pred,max_value_pred+1), alpha=0.7, edgecolor='black', label='Prediction', color='blue')
plt.hist(y_pred[:,0], bins=np.arange(min_value_actual,max_value_actual+1), alpha=0.7, edgecolor='black', label='Actual', color='red')
plt.title('Histogram Distribution of Results')
plt.xlabel('y')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#Deploying the Model
x_new_feat = np.array([[2035, 122, 12, 75, -12, 10, 3, 180, 40, 2]])
X_new_feat = scalerX.transform(x_new_feat)
Y_pred_new = model.predict(X_new_feat)
y_pred_new = scalerY.inverse_transform(Y_pred_new)
print('We predict the new song will have a popularity score of: ' , \
y_pred_new)

#II Artifical Neural Network (Affine Model - Linear, Simplest NN with no Non-Linearity Steps)
linear_model = keras.models.Sequential() #Specify the model
linear_model.add(keras.layers.Dense(1, input_dim=X_train.shape[1], activation=None)) 
linear_model.compile(loss='mse', optimizer='adam', metrics=['mse']) #Specify the loss function, optimisation method and performance metric
history = linear_model.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=500, batch_size=64) #Train model

#Results of optimisation
plt.semilogy(history.history['mse']) #Plot how the model performed during training
plt.semilogy(history.history['val_mse'])
plt.title('Model performance during training')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Prediction
Y_predNN = linear_model.predict(X_test)
y_predNN = scalerY.inverse_transform(Y_predNN)
RMSENN = np.sqrt(np.mean((y_predNN-y_test)**2))
print("RMSE in popularity from NN: " + str(RMSENN))

#On new data
Y_pred_new_NN = linear_model.predict(X_new_feat)
y_pred_new_NN = scalerY.inverse_transform(Y_pred_new_NN)
print('We predict the new song will have a popularity score of: ' , \
y_pred_new_NN)

#Non-Linear Neural Network Model
#3 linear (affine) layers, first two layers use ReLU maps, no activation for last layer
modelNN = keras.models.Sequential()
modelNN.add(keras.layers.Dense(10, input_dim=X_train.shape[1], activation='relu'))
modelNN.add(keras.layers.Dense(50, activation='relu'))
modelNN.add(keras.layers.Dense(1, activation=None))
modelNN.compile(loss='mse', optimizer='adam', metrics=['mse'])
historyNN = modelNN.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=100, batch_size=64) #Train model

#Results of optimisation
plt.semilogy(historyNN.history['mse']) #Plot how the model performed during training
plt.semilogy(historyNN.history['val_mse'])
plt.title('Model performance during training')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

Y_predNN2 = modelNN.predict(X_test)
y_predNN2 = scalerY.inverse_transform(Y_predNN2)
RMSENN2 = np.sqrt(np.mean((y_predNN2-y_test)**2))
print("RMSE in popularity from NN: " + str(RMSENN2))

#On new data
Y_pred_new_NN2 = modelNN.predict(X_new_feat)
y_pred_new_NN2 = scalerY.inverse_transform(Y_pred_new_NN2)
print('We predict the new song will have a popularity score of: ' , \
y_pred_new_NN2)

#Predicting Multiple Variables
x_mv = data[:,[0,1,2,4,5,6,7,8,9]]
y_mv = data[:,[3,10]]

x_train_mv, x_test_mv, y_train_mv, y_test_mv = sklearn.model_selection.train_test_split(x_mv, y_mv, test_size=0.2, random_state=42)

#Scaling Data
scalerX_mv = sklearn.preprocessing.StandardScaler() #Imported Scalers for X
scalerY_mv = sklearn.preprocessing.StandardScaler() #Imported Scalers for Y
scalerX_mv.fit(x_train_mv) #Fitting x training data
scalerY_mv.fit(y_train_mv) #Fitting y training data
#We train the scaling model on the training data but apply it later on the training and test data. Upper case X and Y denote scaled data
X_train_mv = scalerX_mv.transform(x_train_mv)
X_test_mv = scalerX_mv.transform(x_test_mv) 
Y_train_mv = scalerY_mv.transform(y_train_mv)
Y_test_mv = scalerY_mv.transform(y_test_mv)

#Model
model_mv = keras.models.Sequential()
model_mv.add(keras.layers.Dense(10, input_dim=X_train_mv.shape[1], activation='relu'))
model_mv.add(keras.layers.Dense(50, activation='relu'))
model_mv.add(keras.layers.Dense(2, activation=None))
model_mv.compile(loss='mse', optimizer='adam', metrics=['mse'])
history_mv = model_mv.fit(X_train_mv, Y_train_mv, validation_data = (X_test_mv,Y_test_mv), epochs=100, batch_size=64) #Train model

#Results of optimisation
plt.semilogy(history_mv.history['mse']) #Plot how the model performed during training
plt.semilogy(history_mv.history['val_mse'])
plt.title('Model performance during training')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

Y_pred_mv = model_mv.predict(X_test_mv)
y_pred_mv = scalerY_mv.inverse_transform(Y_pred_mv)
RMSE_mv1 = np.sqrt(np.mean((y_pred_mv[0]-y_test_mv[0])**2))
RMSE_mv2 = np.sqrt(np.mean((y_pred_mv[1]-y_test_mv[1])**2))
print("RMSE in danceability from NN_mv: " + str(RMSE_mv1))
print("RMSE in popularity from NN_mv: " + str(RMSE_mv2))

#On new data, deploying the model
x_new_feat_mv = np.array([[2035, 122, 12, -12, 10, 3, 180, 40, 2]])
X_new_feat_mv = scalerX_mv.transform(x_new_feat_mv)
Y_pred_new_mv = model_mv.predict(X_new_feat_mv)
y_pred_new_mv = scalerY_mv.inverse_transform(Y_pred_new_mv)
print('We predict the new song will have a danceability and popularity score of: ' , \
y_pred_new_mv)
