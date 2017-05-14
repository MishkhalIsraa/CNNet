from keras.models import Sequential
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Dense, Dropout, Activation, Flatten
import numpy
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

seed = 100
numpy.random.seed(seed)

dataframe = pd.read_csv('/Users/israamishkhal/Desktop/alldataset.csv')
dataset = dataframe.values
X = dataset[:,0:561]
Y = dataset[:,561]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

s1 = X_train.shape[0]
s2 = X_test.shape[0]
newshape = (s1, 561, 1)
newshape1= (s2,561,1)
x_train = numpy.reshape(X_train, newshape)
x_test =numpy.reshape(X_test, newshape1)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = Sequential()
model.add(Convolution1D(32, 3, border_mode = "same", input_shape = (561, 1)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_length = 2))

model.add(Convolution1D(32, 3, border_mode = "same"))
model.add(Convolution1D(32, 3, border_mode = "same"))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_length = 2))

model.add(Convolution1D(128, 3, border_mode = "same", activation = 'tanh'))
model.add(Convolution1D(128, 3, border_mode = "same", activation = 'tanh'))
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_length = 2))

model.add(Convolution1D(64, 3, border_mode = "same", activation = 'relu'))
model.add(Convolution1D(64, 3, border_mode = "same", activation = 'relu'))
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_length = 2))


model.add(Flatten())
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train,y_train, validation_data=(x_test,y_test), batch_size = 64, nb_epoch = 300)
score = model.evaluate(x_test, y_test, verbose =0)
print("Accuracy: %.2f%%" % (score[1] * 100))
print("Baseline Error: %.2f%%" % (100-score[1]*100))

output = model.predict_classes(x_test)
#output = model.predict(x_test, batch_size=64, verbose=0)
print(output[0:100])


