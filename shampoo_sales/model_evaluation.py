# Silence TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

# Import statements
from __future__ import absolute_import
from pandas import DataFrame, Series, concat, read_csv, datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.externals import joblib
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM
from keras.callbacks import History
from math import sqrt
from matplotlib import pyplot
import numpy
import pickle
import keras

# Data parser
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# Load json and create model
json_file = open('./shampoo_sales/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("./shampoo_sales/model.h5")
print("Loaded model from disk")

# Load scaler object
scaler = joblib.load('./shampoo_sales/lstm_scaler.pkl')

# Load train and test dataframes
train = numpy.genfromtxt('./shampoo_sales/train_df.csv', delimiter=',')
test = numpy.genfromtxt('./shampoo_sales/test_df.csv', delimiter=',')
test = test.reshape(test.shape[0], test.shape[1])
test_scaled = scaler.transform(test)
raw_values = read_csv('./shampoo_sales/shampoo_sales_data.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser).values

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(loaded_model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)-i)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[len(train) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print('Test RMSE: %.3f' % rmse)

# line plot of observed vs predicted
pyplot.plot(raw_values[-12:])
pyplot.plot(predictions)
pyplot.show()

# Evaluate underfitting/overfitting
pyplot(loaded_model.history)
pyplot(lstm_model.history[''])
