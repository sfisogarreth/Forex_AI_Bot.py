import talib as ta  # WHAT: Importing the technical analysis library. WHY: To calculate indicators like RSI. NEXT: We use it for math.
import numpy as np  # WHAT: Importing numerical python. WHY: It handles lists of numbers very fast. NEXT: To clean data formats.
import pandas as pd # WHAT: Importing the "Excel" of Python. WHY: To store our price data in tables. NEXT: To organize indicators.
import datetime as dt # WHAT: Importing time tools. WHY: To handle dates and timestamps. NEXT: To track when prices happened.
import matplotlib.pyplot as plt # WHAT: Importing the graphing tool. WHY: To see our data visually. NEXT: To spot patterns.
import MetaTrader5 as mt5 # WHAT: Importing the broker link. WHY: To eventually place real trades. NEXT: Connecting to your account.
import yfinance as yf # WHAT: Importing the data downloader. WHY: To get free historical price data. NEXT: Training the AI.
import time # WHAT: Importing a clock. WHY: To tell the bot to wait between checks. NEXT: To prevent the bot from crashing.

# We are using Yahoo Finance to get our "History Book" (data). 
# We will use this history to teach the AI what a "Good Trade" looks like before we go live on MT5.

# WHAT: Downloading EUR/USD prices. WHY: We need past prices to find patterns. NEXT: This becomes our "eurusd" table.
eurusd = yf.download("EURUSD=X", start="2020-01-01", end="2026-01-01")

# WHAT: Printing the last 5 rows. WHY: To double-check the data downloaded correctly. NEXT: Seeing the most recent price.
print(eurusd.tail())

# WHAT: Calculating 100-day Simple Moving Average. WHY: To see the long-term trend. NEXT: Comparing this to the current price.
eurusd["SMA_100"] = ta.SMA(eurusd["Close"].to_numpy().ravel(), timeperiod=100)

# WHAT: Calculating the RSI (Tiredness Meter). WHY: To see if people over-bought the Euro. NEXT: Deciding if it's too expensive to buy.
eurusd["RSI"] = ta.RSI(eurusd["Close"].to_numpy().ravel(), timeperiod=14)   

# WHAT: Calculating MACD (Momentum). WHY: To see how fast the price is moving. NEXT: Spotting if a trend is gaining strength.
eurusd["MACD"], eurusd["MACD_signal"], eurusd["MACD_hist"] = ta.MACD(eurusd["Close"].to_numpy().ravel())

# WHAT: Checking the "Shape" of the data. WHY: To ensure we have enough rows for the AI to learn. NEXT: Preparing the charts.
print(eurusd["Close"].values.shape)

# WHAT: Creating a blank canvas (12x8 size). WHY: To make the charts big enough to read. NEXT: Splitting the canvas into rows.
plt.figure(figsize=(12,8))

# WHAT: Creating the TOP box (Row 2, Col 1, Position 1). WHY: To separate Price from RSI. NEXT: Drawing the price line.
plt.subplot(2,1,1)

# WHAT: Drawing the real price. WHY: This is our main focus. NEXT: Overlaying the SMA 100 trend line.
plt.plot(eurusd["Close"], label="Close Price")

# WHAT: Drawing the SMA 100 line. WHY: To see if the price is above or below average. NEXT: Labeling the chart.
plt.plot(eurusd["SMA_100"], label="SMA 100")

# WHAT: Adding a legend. WHY: So we know which line is which. NEXT: Moving to the bottom box.
plt.legend()

# WHAT: Creating the BOTTOM box (Row 2, Col 1, Position 2). WHY: RSI has its own scale (0-100). NEXT: Drawing RSI.
plt.subplot(2,1,2)

# WHAT: Drawing the RSI line in orange. WHY: To make it stand out from the price chart. NEXT: Adding limit lines.
plt.plot(eurusd["RSI"], label="RSI", color="orange")

# WHAT: Drawing a red dashed line at 70. WHY: This marks the "Danger/Expensive" zone. NEXT: Adding a green line.
plt.axhline(70, color="red", linestyle="--")

# WHAT: Drawing a green dashed line at 30. WHY: This marks the "Cheap/Bargain" zone. NEXT: Showing the legend.
plt.axhline(30, color="green", linestyle="--")

# WHAT: Adding the legend for the RSI box. WHY: To confirm the orange line is RSI. NEXT: Popping up the window.
plt.legend()

# WHAT: Final command to show the plot. WHY: Python hides the graph until you ask for it. NEXT: Analyzing the results!
plt.show()