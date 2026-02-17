#this bot is created in the structure of Data -> Indicators -> Brain -> Prediction -> Action
import talib as ta  # WHAT: Technical analysis library. WHY: To calculate indicators like RSI. NEXT: We use it for math.
import numpy as np  # WHAT: Numerical python. WHY: It handles lists of numbers very fast. NEXT: To clean data formats.
import pandas as pd # WHAT: "Excel" of Python. WHY: To store our price data in tables. NEXT: To organize indicators.
import datetime as dt # WHAT: Time tools. WHY: To handle dates and timestamps. NEXT: To track when prices happened.
import MetaTrader5 as mt5 # WHAT: Broker link. WHY: To eventually place real trades. NEXT: Connecting to your account.
import yfinance as yf # WHAT: Data downloader. WHY: To get free historical price data. NEXT: Training the AI.
import time # WHAT: A clock. WHY: To tell the bot to wait between checks. NEXT: To prevent the bot from crashing.

# --- SECTION 1: THE TRADING HANDS ---
def send_trade_order(symbol, type, price, sl, tp):
    # IT PEER NOTE: This handles the "Unsupported Filling Mode" error.
    # We force FOK (Fill or Kill) because most Demo accounts require it.
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01, 
        "type": type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 123456,
        "comment": "AI Bot Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK, # <--- Fixed: Force FOK mode.
    }
    
    result = mt5.order_send(request)
    
    # IT PEER NOTE: If FOK fails, we try IOC (Immediate or Cancel) as a failover.
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        request["type_filling"] = mt5.ORDER_FILLING_IOC
        result = mt5.order_send(request)
        
    return result

# --- SECTION 6: THE AUTOMATION LOOP ---
print("Good Morning :: Gary_Mazi   >>>>>>Robot is now active and watching the clock...")

while True:
    # WHAT: Get the current time.
    now = dt.datetime.now()

    # WHY: Forex markets are closed on Saturdays and Sundays. 
    # now.weekday() < 5 means Monday(0) to Friday(4).
    if now.weekday() < 5: 
        
        # WHY: Testing at 12:45 PM. 
        # NEXT: It will run the brain and send the order once per day.
        if now.hour == 8 and now.minute == 30:
            print(f"It is {now.hour}:{now.minute}. Time to analyze the market!")

            # --- SECTION 2: DATA & INDICATORS ---
            # We are using Yahoo Finance to get our "History Book" (data). 
            data = yf.download("EURUSD=X", start="2020-01-01", end="2026-01-01", auto_adjust=True)
            
            if data.empty:
                print("CRITICAL: Download failed!")
                time.sleep(60) 
                continue 

            # IT PEER NOTE: Flattens MultiIndex columns from yfinance.
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            eurusd = data.copy()

            # Calculating indicators. WHY: To find patterns. NEXT: AI uses these as clues.
            eurusd["SMA_100"] = ta.SMA(eurusd["Close"].to_numpy().ravel(), timeperiod=100)
            eurusd["RSI"] = ta.RSI(eurusd["Close"].to_numpy().ravel(), timeperiod=14)   
            eurusd["upper"], eurusd["middle"], eurusd["lower"] = ta.BBANDS(eurusd["Close"].to_numpy().ravel(), timeperiod=20)
            eurusd["ATR"] = ta.ATR(eurusd["High"].to_numpy().ravel(), 
                                   eurusd["Low"].to_numpy().ravel(), 
                                   eurusd["Close"].to_numpy().ravel(), timeperiod=14)
            eurusd["MACD"], eurusd["MACD_signal"], eurusd["MACD_hist"] = ta.MACD(eurusd["Close"].to_numpy().ravel())

            # --- SECTION 3: CLEANING & TRAINING ---
            # Remove empty rows. WHY: AI needs complete data.
            eurusd = eurusd.dropna()

            # Target: 1 if price goes up tomorrow, 0 if down.
            eurusd["Target"] = np.where(eurusd["Close"].shift(-1) > eurusd["Close"], 1, 0)
            eurusd = eurusd.dropna()

            X = eurusd[["SMA_100", "RSI", "MACD", "MACD_signal", "upper", "lower"]]
            y = eurusd["Target"]

            # Split into training (80%) and testing (20%).
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Random Forest: Like a team of robots arguing to find the best decision.
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # --- SECTION 4: LIVE PREDICTION & RISK ---
            # Predict based on the very latest price data.
            latest_data = X.tail(1)
            prediction = model.predict(latest_data)

            # IT PEER NOTE: .item() turns the "Series" into a single number to prevent f-string errors.
            last_price = eurusd["Close"].iloc[-1].item()
            current_atr = eurusd["ATR"].iloc[-1].item()

            # Risk 1 ATR for SL, seek 1.5 ATR for TP.
            stop_loss_dist = current_atr * 1.0  
            take_profit_dist = current_atr * 1.5

            # --- SECTION 5: FINAL ACTION (MT5) ---
            path_to_mt5 = r"C:\Program Files\MetaTrader 5\terminal64.exe"

            if not mt5.initialize(path=path_to_mt5):
                print(f"MT5 initialize() failed")
            else:
                symbol = "EURUSD" 

                if prediction[0] == 1: # BUY LOGIC
                    price = mt5.symbol_info_tick(symbol).ask
                    sl = price - stop_loss_dist # SL is BELOW for a Buy.
                    tp = price + take_profit_dist # TP is ABOVE for a Buy.
                    result = send_trade_order(symbol, mt5.ORDER_TYPE_BUY, price, sl, tp)
                else: # SELL LOGIC
                    price = mt5.symbol_info_tick(symbol).bid
                    sl = price + stop_loss_dist # SL is ABOVE for a Sell.
                    tp = price - take_profit_dist # TP is BELOW for a Sell.
                    result = send_trade_order(symbol, mt5.ORDER_TYPE_SELL, price, sl, tp)
                
                # IT PEER NOTE: This prints the server's reply (e.g. "Done" or "Invalid Stops").
                print(f"MT5 Response: {result.comment}")
                print(f"AI Decision: {'BUY' if prediction[0] == 1 else 'SELL'} | ENTRY: {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
                
                mt5.shutdown()

            print("Daily Task Completed. Waiting for tomorrow...")
            time.sleep(65) # Wait to avoid double-trading.
    
    time.sleep(30) # Check the clock every 30 seconds.