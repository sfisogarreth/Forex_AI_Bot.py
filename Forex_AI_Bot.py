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
TIMEFRAME = mt5.TIMEFRAME_D1 

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
        "comment": "Smart AI Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK, # WHAT: Fill or Kill. WHY: Fast execution.
    }
    
    result = mt5.order_send(request)
    
    # FAILSAFE: If FOK fails, try IOC (Immediate or Cancel).
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        request["type_filling"] = mt5.ORDER_FILLING_IOC
        result = mt5.order_send(request)
        
    return result

# --- SECTION 4: THE BRAIN (FEATURE ENGINEERING) ---
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
print(f"Morning Mr Mazivanhanga ::: System Online :: {SYMBOL} Strategy Active. Waiting for The Set time ...")

while True:
    now = dt.datetime.now()
    
    # WHAT: Check if it's a weekday (Mon=0, Sun=6).
    # WHY: Forex is closed weekends.
    if now.weekday() < 5: 
        
        # WHAT: Check exact time.
        # WHY: We scan once a day to avoid over-trading.
        if now.hour == 17 and now.minute == 32:
            print("\n--- ANALYZING MARKET ---")
            
            # STEP A: GET DATA
            # WHAT: Download last 2 years of data.
            # WHY: AI needs history to learn patterns.
            data = yf.download("EURUSD=X", period="2y", interval="1d", auto_adjust=True, progress=False)
            
            if data.empty:
                print("Error: No data fetched. Retrying tomorrow.")
                time.sleep(65)
                continue

            # CLEANUP: Fix column format if yfinance gives us complex headers.
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # STEP B: CALCULATE INTELLIGENCE
            # WHAT: Run the `prepare_data` function we wrote above.
            df = prepare_data(data)
            
            # STEP C: DEFINE TARGET & TRAIN
            # WHAT: Create the "Target" column. 
            # LOGIC: If tomorrow's close > today's close, Target = 1 (Buy). Else 0 (Sell).
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
            
            # STEP D: LIVE PREDICTION
            # WHAT: Get the very last row (today's data).
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
            
            print(f"Current ADX: {current_adx:.2f} (Trend Strength)")
            print(f"AI Confidence: SELL {probability[0]*100:.1f}% | BUY {probability[1]*100:.1f}%")

            # STEP E: EXECUTION LOGIC (The Filters)
            
            # FILTER 1: THE CHOPPY MARKET FILTER
            # LOGIC: If ADX is below 25, the market is moving sideways. Do not trade.
            if current_adx > ADX_THRESHOLD:
                
                mt5.initialize()
                
                # FILTER 2: THE CONFIDENCE FILTER
                # LOGIC: Only trade if confidence is > 60%.
                if probability[1] > CONFIDENCE_THRESHOLD:
                    print(">>> SIGNAL: STRONG BUY (High Confidence)")
                    
                    # Risk Management: SL is 1.5x ATR, TP is 2.0x ATR.
                    sl = current_price - (current_atr * 1.5)
                    tp = current_price + (current_atr * 2.0) 
                    
                    send_trade_order(SYMBOL, mt5.ORDER_TYPE_BUY, mt5.symbol_info_tick(SYMBOL).ask, sl, tp)
                    
                elif probability[0] > CONFIDENCE_THRESHOLD:
                    print(">>> SIGNAL: STRONG SELL (High Confidence)")
                    
                    sl = current_price + (current_atr * 1.5)
                    tp = current_price - (current_atr * 2.0)
                    
                    send_trade_order(SYMBOL, mt5.ORDER_TYPE_SELL, mt5.symbol_info_tick(SYMBOL).bid, sl, tp)
                    
                else:
                    # WHAT: Confidence was between 40% and 60%.
                    # WHY: Too risky. We sit on our hands.
                    print(">>> SIGNAL: WEAK / UNCERTAIN. No Trade.")
                
                mt5.shutdown()
            else:
                # WHAT: ADX was < 25.
                # WHY: Market is flat. Strategies fail here.
                print(">>> MARKET CHOPPY (Low ADX). Staying safe (No Trade).")

            print("Analysis Complete. Sleeping...")
            time.sleep(65) # WHAT: Wait 65s. WHY: Ensures we don't trigger twice in the same minute.
            
    time.sleep(30) # WHAT: Heartbeat. WHY: Checks the clock every 30s.