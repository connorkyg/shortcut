import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import statsmodels.api as smapi
from math import sqrt

# Load the data
df = pd.read_csv('bitcoin_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Use the 'Close' column for forecasting
data = df['Close']

# Split the data into training and testing data
train_data = data[:int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]

# Define the ARIMA model
# model = ARIMA(train_data, order=(5, 1, 0))
model = smapi.tsa.arima.ARIMA(train_data, order=(5, 1, 0))

# Fit the model
model_fit = model.fit()

# Make predictions
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Calculate RMSE
rmse = sqrt(mean_squared_error(test_data, predictions))
print('Test RMSE: %.3f' % rmse)
