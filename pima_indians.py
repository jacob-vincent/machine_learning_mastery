# Import statements
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import h5py

# Fix random seed
np.random.seed(42)

# Load dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]

# Train/test split
test = float(input('Enter proportion of data (as a decimal) to use as the test set: '))
train = float(input('Enter the proportion of data (as a decimal) to use as the training set: '))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test,\
 train_size = train, random_state = 42)
np.savetxt('X_train.csv', X_train, delimiter = ',')
np.savetxt('X_test.csv', X_test, delimiter = ',')
np.savetxt('y_train.csv', y_train, delimiter = ',')
np.savetxt('y_test.csv', y_test, delimiter = ',')

# Define model
model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
#model.add(Dense(200, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# Compile model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Fit model
model.fit(X_train, y_train, epochs = 150, batch_size = 10)

# Save model
model.save('model_1.h5')

# Evaluate model
scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
