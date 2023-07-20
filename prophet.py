from fbprophet import Prophet
import pandas as pd

# Load the data
df = pd.read_csv('bitcoin_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

# Define the model
model = Prophet()

# Fit the modelA
model.fit(df)

# Define a dataframe for future predictions
future = model.make_future_dataframe(periods=365)

# Make predictions
forecast = model.predict(future)

# Plot the predictions
model.plot(forecast)
