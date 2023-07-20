import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from datetime import datetime

# Load the data
data = pd.read_csv("/mnt/data/bitcoin_data_period='1d', interval='1m', start='2023-7-11', end='2023-7-18'.csv")

# Preprocess the data
# For simplicity, we'll only use the 'Close' price for prediction
data = data['Close'].values
data = data.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Create the training data
X_train = []
y_train = []

for i in range(60, len(train_data)):
    X_train.append(train_data[i - 60:i])
    y_train.append(train_data[i])

X_train, y_train = np.array(X_train), np.array(y_train)

# Create the LSTM model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Make predictions on test data
x_test = []

for x in range(60, len(test_data)):
    x_test.append(test_data[x - 60:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the test predictions
plt.plot(data[train_size:], color="black", label="Actual Bitcoin Price")
plt.plot(predicted_prices, color="green", label="Predicted Bitcoin Price")
plt.title("Bitcoin Price Prediction")
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()

# Predict next 15 days
prediction_days = 15
temp_input = list(test_data[-60:])
lst_output = []
i = 0
while (i < prediction_days):

    if (len(temp_input) > 60):
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape((1, -1, 1))
        yhat = model.predict(x_input)
        temp_input.append(yhat[0][0])
        temp_input = temp_input[1:]
        lst_output.append(yhat[0][0])
        i = i + 1
    else:
        x_input = temp_input[i:].reshape((1, -1, 1))
        yhat = model.predict(x_input)
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i = i + 1

lst_output = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))
lst_output