#this bot is created in the structure of Data -> Indicators -> Brain -> Prediction -> Action
import talib as ta  #  Importing the technical analysis library. WHY: To calculate indicators like RSI. NEXT: We use it for math.
import numpy as np  # Importing numerical python. WHY: It handles lists of numbers very fast. NEXT: To clean data formats.
import pandas as pd # Importing the "Excel" of Python. WHY: To store our price data in tables. NEXT: To organize indicators.
import datetime as dt #  Importing time tools. WHY: To handle dates and timestamps. NEXT: To track when prices happened.
import matplotlib.pyplot as plt #Importing the graphing tool. WHY: To see our data visually. NEXT: To spot patterns.
import MetaTrader5 as mt5 #Importing the broker link. WHY: To eventually place real trades. NEXT: Connecting to your account.
import yfinance as yf #Importing the data downloader. WHY: To get free historical price data. NEXT: Training the AI.
import time #Importing a clock. WHY: To tell the bot to wait between checks. NEXT: To prevent the bot from crashing.

# --- SECTION 1: THE TRADING HANDS ---
def send_trade_order(symbol, type, price, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.1, 
        "type": type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 123456,
        "comment": "AI Bot Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return mt5.order_send(request)

# --- SECTION 6: THE AUTOMATION LOOP ---
print("Robot is now active and watching the clock...")

while True:
    # WHAT: Get the current time.
    now = dt.datetime.now()

    # WHY: Forex markets are closed on Saturdays and Sundays. 
    # now.weekday() < 5 means Monday(0) to Friday(4).
    if now.weekday() < 5: 
        
        # WHY: We only want the bot to trade at a specific time (e.g., 08:00 AM).
        # NEXT: It will run the brain and send the order once per day.
        if now.hour == 8 and now.minute == 0:
            print(f"It is {now.hour}:{now.minute}. Time to analyze the market!")

            # --- SECTION 2: DATA & INDICATORS ---
            # We are using Yahoo Finance to get our "History Book" (data). 
            # We will use this history to teach the AI  a "Good Trade" looks like before we go live on MT5.

            #by me downloading EUR/USD prices(and so far am using this one pair) WHY: We need past prices to find patterns. NEXT: This becomes our "eurusd" table.
            eurusd = yf.download("EURUSD=X", start="2020-01-01", end="2026-01-01")

            #Printing the last 5 rows. WHY: To double-check the data downloaded correctly. NEXT: Seeing the most recent price.
            print(eurusd.tail())

            #Calculating 100-day Simple Moving Average. WHY: To see the long-term trend. NEXT: Comparing this to the current price.
            eurusd["SMA_100"] = ta.SMA(eurusd["Close"].to_numpy().ravel(), timeperiod=100)

            #Calculating the RSI (Tiredness Meter). WHY: To see if people over-bought the Euro. NEXT: Deciding if it's too expensive to buy.
            eurusd["RSI"] = ta.RSI(eurusd["Close"].to_numpy().ravel(), timeperiod=14)   

            eurusd["upper"], eurusd["middle"], eurusd["lower"] = ta.BBANDS(eurusd["Close"].to_numpy().ravel(), timeperiod=20)

            eurusd["ATR"] = ta.ATR(eurusd["High"].to_numpy().ravel(), 
                                   eurusd["Low"].to_numpy().ravel(), 
                                   eurusd["Close"].to_numpy().ravel(), timeperiod=14)

            #Calculating MACD (Momentum). WHY: To see how fast the price is moving. NEXT: Spotting if a trend is gaining strength.
            eurusd["MACD"], eurusd["MACD_signal"], eurusd["MACD_hist"] = ta.MACD(eurusd["Close"].to_numpy().ravel())

            # --- SECTION 3: CLEANING & TRAINING ---
            #first thing i have to do is to remove the first 100 rows because the SMA 100 will be empty for those. WHY: The AI needs complete data to learn. NEXT: Cleaning the dataset.
            eurusd = eurusd.dropna()

            #now for the target we just have to use 0 or 1 the language that A understand such that we take the predictad price of tomoe and then we compare to the todays price
            #if tmoe priice is greater than today then today we can buy else we sell thats the code bellow this also will have its own column
            #we also used the np.where since it allows us to look at all rows at once than use iterating each row using for loop which can be slow if we have a large data np.where( CONDITION , DO_THIS_IF_TRUE , DO_THIS_IF_FALSE )
            eurusd["Target"] = np.where(eurusd["Close"].shift(-1) > eurusd["Close"], 1, 0)
            # We dropna one more time to remove the last row created by the shift
            eurusd = eurusd.dropna()

            #now we have our target column ready and we can now start to train our model using the indicators as input and the target as output
            #we will use the RSI and the SMA 100 as our indicators for now but we can always add more later on as we get more comfortable with the process. WHY: These indicators are commonly used to predict price movements. NEXT: Preparing the data for the AI.
            X = eurusd[["SMA_100", "RSI", "MACD", "MACD_signal", "upper", "lower"]]
            y = eurusd["Target"]

            #to split the data into train and testing with the 0.2 and 0.8 ratio 
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            #now we have our training and testing data ready we can now start to train our model using a simple algorithm like random forest clasification
            from sklearn.ensemble import RandomForestClassifier#this part is more like the team of robots argueing if we should buy or sell and then the most valuable decesion wins
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # --- SECTION 4: LIVE PREDICTION & RISK ---
            # WHAT: Get the very last row of your data (Today). To see what the AI thinks will happen tomorrow.
            # This is the signal you would actually use to trade.
            latest_data = X.tail(1)
            prediction = model.predict(latest_data)

            # FIX: We define these variables HERE so the computer knows them before the next steps!
            last_price = eurusd["Close"].iloc[-1]
            current_atr = eurusd["ATR"].iloc[-1]

            # 3. Calculate TP and SL (Risk Management)
            # We risk 1 ATR to make 1.5 ATR (The 1:1.5 Ratio)
            stop_loss_dist = current_atr * 1.0  
            take_profit_dist = current_atr * 1.5

            # --- SECTION 5: FINAL ACTION (MT5) ---
            # WHAT: The "Home Address" of your MT5 program.
            path_to_mt5 = r"C:\Program Files\MetaTrader 5\terminal64.exe"

            # WHY: We must put 'path=path_to_mt5' inside the parentheses so Python knows where to go.
            # NEXT: If it works, it will open MT5 and log you in.
            if not mt5.initialize(path=path_to_mt5):
                print(f"MT5 initialize() failed, error code = {mt5.last_error()}")
            else:
                # --- CONNECTION VERIFICATION ---
                # WHAT: Requesting account information from the terminal. prove bridge is open.
                account_info = mt5.account_info()

                if account_info is not None:
                    print("-" * 30)
                    print(f"LINK VERIFIED! Connected to Account: {account_info.login}")
                    print(f"Current Balance: ${account_info.balance}")
                    print("-" * 30)
                else:
                    print("LINK FAILED: Python is running, but cannot talk to MT5.")

                symbol = "EURUSD" # Adjust this to match your MT5 Market Watch!

                if prediction[0] == 1:
                    price = mt5.symbol_info_tick(symbol).ask
                    tp = last_price + take_profit_dist
                    sl = last_price - stop_loss_dist
                    result = send_trade_order(symbol, mt5.ORDER_TYPE_BUY, price, sl, tp)
                    print(f"AI Order Sent: BUY | ENTRY: {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
                else:
                    price = mt5.symbol_info_tick(symbol).bid
                    tp = last_price - take_profit_dist
                    sl = last_price + stop_loss_dist
                    result = send_trade_order(symbol, mt5.ORDER_TYPE_SELL, price, sl, tp)
                    print(f"AI Order Sent: SELL | ENTRY: {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
                
                # Always shut down the connection when done to save computer resources
                mt5.shutdown()

            print("Daily Task Completed. Waiting for tomorrow...")
            
            # NOTE: After trading, we sleep for 65 seconds so it doesn't 
            # trade 100 times in the same minute!
            time.sleep(65) 
    
    # WHAT: Wait 30 seconds before checking the clock again.
    # WHY: To prevent the CPU from running at 100% usage.
    time.sleep(30)