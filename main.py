import pandas as pd
import yfinance as yf

# Define the ticker symbol for Bitcoin
tickerSymbol = 'BTC-USD'

# Get the data
tickerData = yf.Ticker(tickerSymbol)

# Get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', interval='1m', start='2023-7-11', end='2023-7-18')

# Save the data to a csv file
tickerDf.to_csv('bitcoin_data_tmp.csv')


#             period : str
#                 Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#                 Either Use period parameter or use start and end
#             interval : str
#                 Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
#                 Intraday data cannot extend last 60 days