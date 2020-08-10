import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM

#Creating 1D Data into Time-Series

def newdataset(dataset, step_size):
	Xdata, Ydata = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size), 0]
		Xdata.append(a)
		Ydata.append(dataset[i + step_size, 0])
	return np.array(Xdata), np.array(Ydata)

#Loading Dataset

datasetfile = pd.read_csv("Bitcoin.csv")

#Converting date column to datetime

datasetfile['Date'] = pd.to_datetime(datasetfile['Date'])

#Reindexing dataset by Date column

datasetfile = datasetfile.reindex(index= datasetfile.index[::-1])
zaman = np.arange(1, len(datasetfile) + 1, 1)
OHCL_avg = datasetfile.mean(axis=1)
plt.plot(zaman, OHCL_avg)
plt.title("Plot between Reindexed Average OHCL values")
plt.xlabel('Time', fontsize=13)
plt.ylabel('Price', fontsize=13)
plt.show()

#Normalizing dataset

OHCL_avg = np.reshape(OHCL_avg.values, (len(OHCL_avg),1))
scaler = MinMaxScaler(feature_range=(0,1))
OHCL_avg = scaler.fit_transform(OHCL_avg)

#Training and Testing dataset

OHLCtrain = int(len(OHCL_avg)*0.56)
OHLCtest = len(OHCL_avg) - OHLCtrain
OHLCtrain , OHLCtest = OHCL_avg[0:OHLCtrain,:], OHCL_avg[OHLCtrain:len(OHCL_avg),:]

#Creating 1dataset from mean OHLV

Xtrain, Ytrain = newdataset(OHLCtrain,1)
Xtest, Ytest = newdataset(OHLCtest,1)

# Reshape dataset for LSTM in 3D Dimension
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0],1,Xtrain.shape[1]))
Xtest = np.reshape(Xtest, (Xtest.shape[0],1,Xtest.shape[1]))
step_size = 1

#LSTM Model created

model = Sequential()
model.add(LSTM(128, input_shape=(1, step_size)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(Xtrain, Ytrain, epochs=10, batch_size=25, verbose=2)
trainPredict = model.predict(Xtrain)
testPredict = model.predict(Xtest)

#De-Normalizing for plotting 

trainPredict = scaler.inverse_transform(trainPredict)
Ytrain = scaler.inverse_transform([Ytrain])
testPredict = scaler.inverse_transform(testPredict)
Ytest = scaler.inverse_transform([Ytest])

#Performance Measure RMSE is calculated for predicted train dataset

trainScore = math.sqrt(mean_squared_error(Ytrain[0], trainPredict[:,0]))
print("\nTrain RMSE: %.2f \n" % (trainScore))

#Performance Measure RMSE is calculated for predicted test dataset

testScore = math.sqrt(mean_squared_error(Ytest[0], testPredict[:,0]))
print("Test RMSE: %.2f \n" % (testScore))

#Converting predicted train dataset for plotting

trainPredictPlot = np.empty_like(OHCL_avg)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[step_size:len(trainPredict)+step_size,:] = trainPredict

#Converting predicted test dataset for plotting

testPredictPlot = np.empty_like(OHCL_avg)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHCL_avg)-1,:] = testPredict


#Visualizing predicted values

OHCL_avg = scaler.inverse_transform(OHCL_avg)
plt.plot(OHCL_avg, 'r', label='Orginal Dataset')
plt.plot(trainPredictPlot, 'g', label='Training DataSet')
plt.plot(testPredictPlot, 'b', label='Predicted price/test set')
plt.title("Predicted Hourly Bitcoin Prices")
plt.xlabel('Hourly Time', fontsize=13)
plt.ylabel('Close Price', fontsize=13)
plt.legend(loc='upper right')
plt.show()