#FEEG6042 - Introduction to Machine Learning Lab 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import keras

df = pd.read_csv('faults.csv')
data = df.to_numpy()

faults = [29,32,33]; #Classify data with three types of faults, this removes data points with different faults
col_selection = np.sum(data[:,faults],axis=1)==1 #Find if there is a 1 in the three columns we want
data = data[col_selection,:] #Choose wanted data

for i in range(3):
    print(len(data[:,faults[i]])) #Number of Datapoints

x = data[:,0:26] #X Training Data
y = data[:,faults] #Y Training Data

C_labels = np.argmax(y,axis=1)

#Histograms
#for i in range(26):
    #min_value = min(x[i,:])
    #max_value = max(x[i,:])
    #bins = np.linspace(min_value, max_value, 100)
    #plt.hist(x[i,:], bins=bins, alpha=0.7, edgecolor='black')
    #plt.show()

#plt.figure()
#for c in range(y.shape[1]):
    #row_idx = y[:,c]==1
    #plt.plot(x[row_idx,i],x[row_idx,j],'.',ms=0.5)
    #plt.title(df.columns[i] + ' vs ' + df.columns[j])

#Splitting Training Data
x_train, x_test, c_train, c_test = sklearn.model_selection.train_test_split(x, C_labels, test_size=0.2, random_state=42)

#Scaling Data
scalerX = sklearn.preprocessing.StandardScaler() #Imported Scalers for X
scalerX.fit(x_train) #Fitting x training data
#We train the scaling model on the training data but apply it later on the training and test data. Upper case X and Y denote scaled data
X_train = scalerX.transform(x_train)
X_test = scalerX.transform(x_test) 

X_train_mean = np.mean(X_train, axis=0) #axis 0 represents columns and axis 1 represents rows
X_test_mean = np.mean(X_test, axis=0)
X_train_std = np.std(X_train, axis=0) 
X_test_std = np.std(X_test, axis=0)

print(f'Xtrain mean: {X_train_mean}') #Print Statements
print(f'Xtrain std: {X_train_std}')
print(f'Xtest mean: {X_test_mean}')
print(f'Xtest std: {X_test_std}')

onehot = sklearn.preprocessing.OneHotEncoder()
onehot.fit(c_train.reshape(-1,1)) # We need to re-shape here as
# OneHotEncoder needs a 2D array,
# not a 1D vector
Y_train = onehot.transform(c_train.reshape(-1,1)).toarray()
Y_test = onehot.transform(c_test.reshape(-1,1)).toarray()

#Training the Model, #Accuracy should be going up, while crossentropy should be going down for a better model
#model = keras.models.Sequential()
#model.add(keras.layers.Dense(5, input_dim=X_train.shape[1], activation='relu'))
#model.add(keras.layers.Dense(10, activation='relu'))
#model.add(keras.layers.Dense(10, activation='relu'))
#model.add(keras.layers.Dense(3, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#history = model.fit(X_train, Y_train, validation_data = (X_test,Y_test),\
#epochs=200, batch_size=64)

#Results of optimisation
#Accuracy
#plt.semilogy(history.history['accuracy']) #Plot how the model performed during training
#plt.semilogy(history.history['val_accuracy'])
#plt.title('Accuracy During Training')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()

#Cross Entropy
#plt.semilogy(history.history['loss']) 
#plt.semilogy(history.history['val_loss'])
#plt.title('Cross Entropy During Training')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()

#Testing the model
#y_pred = model.predict(X_test) #This gives probability that the data belongs to a class
#c_pred = np.argmax(y_pred,axis=1) #Assigns class based on the highest probability

#Confusion Matrix
#C = sklearn.metrics.confusion_matrix(c_test, c_pred,\
#normalize='true')
#sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=C,\
#display_labels=[0,1,2]).plot()
#plt.show()

#Clustering
kmeans = sklearn.cluster.KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_train)
print(kmeans.labels_)

k_pred = kmeans.predict(X_test)
D = sklearn.metrics.confusion_matrix(c_test, k_pred,\
normalize='true')
sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=D,\
display_labels=[0,1,2]).plot()
plt.show()