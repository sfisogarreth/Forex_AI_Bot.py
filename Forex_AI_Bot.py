# --- SECTION 1: IMPORTS & BRAIN TOOLS ---
import talib as ta  # WHAT: Technical Analysis Library. WHY: Calculates RSI, SMA, ADX. NEXT: Math for the bot.
import numpy as np  # WHAT: Number cruncher. WHY: Fast math for arrays. NEXT: Data formatting.
import pandas as pd # WHAT: Data tables. WHY: Holds price history like Excel. NEXT: Organizing indicators.
import datetime as dt # WHAT: Time tool. WHY: Tracks market hours. NEXT: Checking if market is open.
import MetaTrader5 as mt5 # WHAT: The Broker. WHY: Executes the trades. NEXT: Buying/Selling.
import yfinance as yf # WHAT: Data source. WHY: Gets historical data to train the AI. NEXT: Learning.
import time # WHAT: Clock. WHY: Pauses the bot. NEXT: Prevents crashing.

# UPGRADE ALERT: We replaced RandomForest with GradientBoosting.
# WHY: Random Forest is like a vote. Gradient Boosting is like a team that learns from mistakes.
# NEXT: Smarter predictions.
from sklearn.ensemble import GradientBoostingClassifier 

# --- SECTION 2: CONFIGURATION (THE RULES) ---
SYMBOL = "EURUSD"

# LEARNING NOTE: Timeframes
# WHAT: We are trading the 4-Hour chart.
# WHY: 4H trends are more stable than 1H or 15M. Less "noise" (random movement).
TIMEFRAME = mt5.TIMEFRAME_H4 

# UPGRADE ALERT: The "Confidence" Filter.
# WHAT: We only trade if the AI is >60% sure. 
# WHY: The old bot traded at 51% (a coin flip). We want high conviction only.
CONFIDENCE_THRESHOLD = 0.60 

# UPGRADE ALERT: The "Choppy Market" Filter.
# WHAT: Average Directional Index (ADX).
# WHY: If ADX < 25, the market is flat/choppy. We lose money in chop.
# NEXT: We will refuse to trade if ADX is low.
ADX_THRESHOLD = 25 

# --- SECTION 3: THE TRADING HANDS ---
def check_open_positions(symbol):
    # WHAT: Position Checker.
    # WHY: You asked to trade "till it takes profit or stop loss".
    # LOGIC: If we already have a trade open, we MUST NOT open another one.
    # ALTERNATIVE: We could "pyramid" (add more trades), but that is high risk.
    positions = mt5.positions_get(symbol=symbol)
    
    if positions is None or len(positions) == 0:
        return False # No positions open, safe to trade.
    
    return True # Position exists, HOLD FIRE.

def send_trade_order(symbol, type_trade, price, sl, tp):
    # WHAT: The order package.
    # WHY: MT5 needs specific details to accept a trade.
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01,
        "type": type_trade,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 234000, # WHAT: ID number. WHY: Lets us track *this* bot's trades.
        "comment": "Smart AI H4 Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK, # WHAT: Fill or Kill. WHY: Fast execution.
    }
    
    result = mt5.order_send(request)
    
    # FAILSAFE: If FOK fails, try IOC (Immediate or Cancel).
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        request["type_filling"] = mt5.ORDER_FILLING_IOC
        result = mt5.order_send(request)
        
    return result

# --- SECTION 4: THE BRAIN (DATA PROCESSING) ---

def resample_to_4h(df_1h):
    # LEARNING NOTE: Resampling (The "Magic" Trick)
    # WHAT: Turning 1-hour candles into 4-hour candles.
    # WHY: Yahoo Finance gives great 1H data, but sometimes 4H data is missing or messy.
    # HOW IT WORKS:
    #   - Open price = The Open of the 1st hour.
    #   - High price = The Highest price of the 4 hours.
    #   - Low price = The Lowest price of the 4 hours.
    #   - Close price = The Close of the 4th hour.
    
    # Logic: Group by 4 Hours
    aggregation = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    # Execute the math
    df_4h = df_1h.resample('4h').agg(aggregation)
    
    # Cleanup: Remove the last row if the 4 hours aren't finished yet.
    df_4h.dropna(inplace=True)
    return df_4h

def prepare_data(df):
    # WHAT: Adding indicators.
    # WHY: Raw price isn't enough. The AI need "features" (clues) to learn.
    
    # 1. Trend Indicators
    df["SMA_50"] = ta.SMA(df["Close"], timeperiod=50) # Medium trend
    df["SMA_200"] = ta.SMA(df["Close"], timeperiod=200) # Long term trend
    
    # 2. Oscillators (Overbought/Oversold)
    df["RSI"] = ta.RSI(df["Close"], timeperiod=14)
    
    # 3. Volatility (How much does price move?)
    df["ATR"] = ta.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
    
    # UPGRADE ALERT: ADX (Trend Strength)
    # WHAT: Measures *how strong* the trend is, not just direction.
    # WHY: Tells us to stay away if the market is sleeping.
    df["ADX"] = ta.ADX(df["High"], df["Low"], df["Close"], timeperiod=14)
    
    # UPGRADE ALERT: Lag Features (Momentum)
    # WHAT: "What was RSI yesterday?"
    # WHY: Knowing *change* is better than knowing *value*. 
    # Example: RSI 60 is okay. RSI moving from 30 to 60 is BULLISH.
    df["RSI_Lag1"] = df["RSI"].shift(1)
    df["Close_Lag1"] = df["Close"].shift(1)
    
    # Cleanup: Remove rows with empty values (NaN) created by indicators.
    df.dropna(inplace=True)
    return df

# --- SECTION 5: THE STRATEGY LOOP ---
print(f"Morning Mr Mazivanhanga ::: System Online :: {SYMBOL} H4 Strategy Active.")
print("Waiting for top of the hour (Minute :00)...")

while True:
    now = dt.datetime.now()
    
    # WHAT: Check if it's a weekday (Mon=0, Sun=6).
    # WHY: Forex is closed weekends.
    if now.weekday() < 5: 
        
        # LEARNING NOTE: The Hourly Trigger
        # WHAT: We check `now.minute`.
        # WHY: We want to trade the *close* of the candle.
        # Since we are building 4H candles from 1H data, we check every hour to see
        # if a new 4H block has just finished.
        if 0 <= now.minute <= 2:
            
            # CHECK: Do we have a trade open?
            # LOGIC: If yes, we skip everything. We wait for the Stop Loss or Take Profit to hit.
            mt5.initialize()
            if check_open_positions(SYMBOL):
                print(f"[{now.hour}:{now.minute}] Trade is LIVE. Managing position... (No new entry)")
                mt5.shutdown()
                time.sleep(300) # Sleep 5 mins to let the hour pass.
                continue
            
            print(f"\n--- ANALYZING 4H MARKET at {now.hour}:{now.minute} ---")
            
            # STEP A: GET DATA (1 Hour)
            # WHAT: Download last 60 days of 1H data.
            # WHY: We need lots of 1H data to build enough 4H candles for the AI.
            data = yf.download("EURUSD=X", period="60d", interval="1h", auto_adjust=True, progress=False)
            
            if data.empty:
                print("Error: No data fetched. Retrying next hour.")
                time.sleep(65)
                continue

            # CLEANUP: Fix column format if yfinance gives us complex headers.
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # STEP B: CONVERT TO 4-HOUR (Resampling)
            # WHAT: Run the math function we wrote above.
            df_4h = resample_to_4h(data)

            # STEP C: CALCULATE INTELLIGENCE
            # WHAT: Run the `prepare_data` function on our new 4H data.
            df = prepare_data(df_4h)
            
            # STEP D: DEFINE TARGET & TRAIN
            # WHAT: Create the "Target" column. 
            # LOGIC: If next 4H close > current close, Target = 1 (Buy).
            df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
            
            # WHAT: Select the columns the AI gets to see.
            features = ["SMA_50", "SMA_200", "RSI", "RSI_Lag1", "ADX", "ATR"]
            
            # WHAT: Train on all past data (except the very last candle which has no future yet).
            train_df = df.iloc[:-1]
            X = train_df[features]
            y = train_df["Target"]
            
            # UPGRADE: Gradient Boosting
            # WHAT: A stronger learning model than Random Forest.
            # WHY: It builds trees sequentially, correcting errors from the previous tree.
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            model.fit(X, y)
            
            # STEP E: LIVE PREDICTION
            # WHAT: Get the very last row (the 4H candle that just closed).
            current_data = df.iloc[[-1]][features]
            
            # UPGRADE: PROBABILITY CHECK
            # WHAT: model.predict_proba() instead of model.predict().
            # WHY: Gives us % confidence (e.g., 65% Buy) instead of just "Buy".
            probability = model.predict_proba(current_data)[0]
            # probability[0] = Chance of Down (Sell)
            # probability[1] = Chance of Up (Buy)
            
            current_adx = current_data["ADX"].values[0]
            current_atr = current_data["ATR"].values[0]
            current_price = df.iloc[-1]["Close"]
            
            print(f"Current ADX (4H): {current_adx:.2f} (Trend Strength)")
            print(f"Bot_Mazi Confidence: SELL {probability[0]*100:.1f}% | BUY {probability[1]*100:.1f}%")

            # STEP F: EXECUTION LOGIC (The Filters)
            
            # FILTER 1: THE CHOPPY MARKET FILTER
            # LOGIC: If ADX is below 25, the market is moving sideways. Do not trade.
            if current_adx > ADX_THRESHOLD:
                
                # FILTER 2: THE CONFIDENCE FILTER
                # LOGIC: Only trade if confidence is > 60%.
                if probability[1] > CONFIDENCE_THRESHOLD:
                    print(">>> SIGNAL: STRONG BUY (High Confidence)")
                    
                    # Risk Management (4H): 
                    # WHAT: 4H candles are big, so Stops must be wider.
                    # SL is 1.5x ATR, TP is 2.5x ATR (Aiming for a big swing).
                    sl = current_price - (current_atr * 1.5)
                    tp = current_price + (current_atr * 2.5) 
                    
                    send_trade_order(SYMBOL, mt5.ORDER_TYPE_BUY, mt5.symbol_info_tick(SYMBOL).ask, sl, tp)
                    
                    # SLEEP: We traded. Now sleep 5 mins to get out of the "Minute 0" window.
                    time.sleep(300)
                    
                elif probability[0] > CONFIDENCE_THRESHOLD:
                    print(">>> SIGNAL: STRONG SELL (High Confidence)")
                    
                    sl = current_price + (current_atr * 1.5)
                    tp = current_price - (current_atr * 2.5)
                    
                    send_trade_order(SYMBOL, mt5.ORDER_TYPE_SELL, mt5.symbol_info_tick(SYMBOL).bid, sl, tp)
                    
                    time.sleep(300)
                    
                else:
                    # WHAT: Confidence was between 40% and 60%.
                    # WHY: Too risky. We sit on our hands.
                    print(">>> Mr Mazi The SIGNAL is weak in (retracement): WEAK / UNCERTAIN. Therefore Dont Trade.")
                
            else:
                # WHAT: ADX was < 25.
                # WHY: Market is flat. Strategies fail here.
                print(">>> MARKET CHOPPY (Low ADX). Staying safe (No Trade).")

            mt5.shutdown()
            print("Analysis Complete. Sleeping...")
            
            # SAFETY SLEEP: Sleep 60s so we don't accidentally run again in the same minute.
            time.sleep(60)
            
    # WHAT: Heartbeat. 
    # WHY: Checks the clock every 30s to see if the next hour has started.
    time.sleep(30)