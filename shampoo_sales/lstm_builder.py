# Silence TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

# Import Statements
#from __future__ import absolute_import
from pandas import DataFrame, Series, concat, read_csv, datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.callbacks import LambdaCallback, History
from math import sqrt
from matplotlib import pyplot
import numpy
import pickle

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	'''scaler = MinMaxScaler(feature_range=(-1, 1))'''
	scaler = StandardScaler()
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled


# fit an LSTM network to training data
def fit_lstm(train, test, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])

	x_test, y_test = test[:, 0:-1], test[:, -1]
	x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

	history = History()

	print(X.shape)
	print(x_test.shape)
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=False, return_sequences=True))
	model.add(Dense(2*neurons, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(neurons, activation='relu'))
	model.add(LSTM(neurons, stateful=True, return_sequences=False))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='mean_squared_error', optimizer='adam')
	h = model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=False, validation_data=(x_test, y_test), callbacks=[history])
	return model, h

# load dataset
series = read_csv('~/Desktop/Machine_learning_mastery/shampoo_sales/shampoo_sales_data.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets, saving results
train, test = supervised_values[0:-12], supervised_values[-12:]
numpy.savetxt('train_df.csv', train, delimiter=',')
numpy.savetxt('test_df.csv', test, delimiter=',')

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# Save scaler object and scaled data
joblib.dump(scaler, 'lstm_scaler.pkl')
numpy.savetxt('train_scaled_df.csv', train_scaled, delimiter=',')
numpy.savetxt('test_scaled_df.csv', test_scaled, delimiter=',')

print(train_scaled.shape, test_scaled.shape)

# fit the model
lstm_model, hist = fit_lstm(train_scaled, test_scaled, 1, 256, 8)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# Save model
# serialize model to JSON
model_json = lstm_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
lstm_model.save_weights("model.h5")
print("Saved model to disk")

with open('model_history.pkl', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)
