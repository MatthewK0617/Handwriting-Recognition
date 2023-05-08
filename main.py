import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data() # loaded data

print(x_train.shape, y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) # reshaped the data
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

y_train = to_categorical(y_train, num_classes=10) # transformed into binary
y_test = to_categorical(y_test, num_classes=10)

x_train = x_train.astype('float32') # convert to a specific type
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


batch_size = 128
num_classes = 10
epochs = 10 # number of times it goes through the NN
l1 = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)
l2 = Conv2D(64, (3, 3), activation='relu')
l3 = MaxPooling2D(pool_size=(2,2))
l4 = Dropout(0.25)
l5 = Flatten()
l6 = Dense(256, activation='relu')
l7 = Dropout(0.5)
l8 = Dense(num_classes, activation='softmax')

model = Sequential() # kernels are layers in NN that allows for higher dimensional analysis
model.add(l1)
model.add(l2)
model.add(l3)
model.add(l4)
model.add(l5)
model.add(l6)
model.add(l7)
model.add(l8)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test)) # fitting (training) model using training data
print("Trained model")

model.save('mnist.h5')
print("Saving model")

score = model.evaluate(x_test, y_test, verbose=0) # testing model
print('Test loss:', score[0])
print('Test accuracy', score[1])
