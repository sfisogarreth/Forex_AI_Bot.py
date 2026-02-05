import talib as ta
import numpy as np
import pandas as pd 
import datetime as dt
import matplotlib.pyplot as plt
import MetaTrader5 as mt5   
import yfinance as yf
import time 


#then for me to atleast get the data range that i willl train instead downloading the data from mt5 i will use yahoo finance 
# to get the data and then i will train the model on that data and then i will use the model to predict the future price of the currency pair and then i will use that prediction to make a trade on my local drive i can use the mt5 to make the trade and then i will use the mt5 to get the data for the next day and then i will use that data to train the model again and then i will use that model to predict the future price of the currency pair and then i will use that prediction to make a trade on my local drive and then i will use the mt5 to get the data for the next day and then i will use that data to train the model again and then i will use that model to predict the future price of the currency pair and then i will use that prediction to make a trade on my local drive and then i will use the mt5 to get the data for the next day and then i will use that data to train the model again and then i will use that model to predict the future price of the currency pair and then i will use that prediction to make a trade on my local drive and then i will use the mt5 to get the data for the next day and then i will use that data to train the model again and then i will use that model to predict the future price of the currency pair and then i will use that prediction to make a trade on my local drive and then i will use the mt5 to get the data for the next day and then i will use that data to train the model again and then i will use that model to predict the future price of the currency pair and then i will use that prediction to make a trade on my local drivei can use the pandas_reader to get the data from yahoo finance and then i can use the ta library to calculate the technical indicators and then i can use the sklearn library to train the model and then i can use the 
# model to predict the future price of the currency pair and then i can use that prediction to make a trade on my local drive and then i can use the mt5 to get the data for the next day and then i can use that data to train the model again and then i can use that model to predict the future price of the currency pair and then i can use that prediction to make a trade on my local drive and then i can use the mt5 to get the data for the next day and then i can use that data to train the model again and then i can use that model to predict the future price of the currency pair and then i can use that prediction to make a trade on my local drive and then i can use the mt5 to get the data for the next day and then i can use that data to train the model again and then i will use that model to predict the future          price of the currency pair and then i will use that prediction to make a trade on my local drivei will also need to implement a stop loss and take profit mechanism in order to manage my risk and also in order to maximize my profits.

eurusd = yf.download("EURUSD=X", start="2020-01-01", end="2026-01-01")
print(eurusd.tail())


eurusd["SMA_100"] = ta.SMA(eurusd["Close"].to_numpy().ravel(), timeperiod=100)
eurusd["RSI"] = ta.RSI(eurusd["Close"].to_numpy().ravel(), timeperiod=14)   # usually RSI uses 14
eurusd["MACD"], eurusd["MACD_signal"], eurusd["MACD_hist"] = ta.MACD(eurusd["Close"].to_numpy().ravel())

print(eurusd["Close"].values.shape)

plt.figure(figsize=(12,8))

# Price + SMA
plt.subplot(2,1,1)
plt.plot(eurusd["Close"], label="Close Price")
plt.plot(eurusd["SMA_100"], label="SMA 100")
plt.legend()

# RSI
plt.subplot(2,1,2)
plt.plot(eurusd["RSI"], label="RSI", color="orange")
plt.axhline(70, color="red", linestyle="--")
plt.axhline(30, color="green", linestyle="--")
plt.legend()

plt.show()