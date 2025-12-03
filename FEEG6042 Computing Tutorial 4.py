#FEEG6042: Computing Tutorial 4
import sklearn
import keras
import keras.datasets.mnist as mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
#Plotting data
for i in range(10):
    x = x_train[(y_train==i).squeeze(),:,:] #Takes data from y classes, squeezes removes dimension size 1, eg (50000, 1) -> (50000,), : takes all data in dimensions
    plt.subplot(2,5,i+1) #Index to location
    plt.imshow(x[0]) #Takes the first image data to display, 32x32 pixels with RGB color channels
    plt.axis('off') #No plot axis
plt.show() #Show

#Scaler
scaler = sklearn.preprocessing.MinMaxScaler()

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

#Model specified without sequential model
input = keras.Input(shape=X_train.shape[1:]) #Input specified as an appropriate shape for the input data

x = input
x = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu')(x)
x = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu')(x)
x = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(100, activation='sigmoid')(x)
x = keras.layers.Dense(10, activation='softmax')(x)
cnn1=keras.Model(input,x)
cnn1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history1 = cnn1.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs=5, batch_size=64)

#Model with two inputs
#Building model that can take image data but also additional features - this model has higher accuracy due to the additional feature
e_test=(y_test%2).reshape((-1,1)).astype('float32')
e_train = (y_train%2).reshape((-1,1)).astype('float32')

inputE = keras.Input(shape=(1,))
inputI = keras.Input(shape=(28,28))
x = inputI
x = keras.layers.Reshape((28,28,1))(x)
x = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu')(x)
x = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu')(x)
x = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Flatten()(x)

# he second branch works on the categorial data
y = inputE
z = keras.layers.Concatenate()([x,y])
z = keras.layers.Dense(100, activation='sigmoid')(z)
z = keras.layers.Dense(10, activation='softmax')(z)
cnn2 = keras.Model(inputs=[inputI, inputE], outputs=z)
cnn2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history2 = cnn2.fit([X_train, e_train], Y_train, validation_data = ([X_test, e_test],Y_test), epochs=5, batch_size=64)

#Results of optimisation
plt.semilogy(history1.history['accuracy']) 
plt.semilogy(history1.history['val_accuracy'])
plt.semilogy(history2.history['accuracy']) 
plt.semilogy(history2.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Model 1 Train', 'Model 1 Test', 'Model 2 Train', 'Model 2 Test'], loc='upper left')
plt.show()