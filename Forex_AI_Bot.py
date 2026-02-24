# --- section 1: imports & brain tools ---
import talib as ta  # bringing in TA-Lib to handle the heavy math for indicators like RSI and ADX so we don't have to write complex formulas from scratch
import numpy as np  # fast number cruncher for huge lists of price data
import pandas as pd # basically an invisible excel spreadsheet inside the code to organize our history
import datetime as dt # time tracker to know when the market is open and when the hour changes
import MetaTrader5 as mt5 # the bridge to the broker to actually pull the trigger on trades
import time # simple pause function to stop the loop from crashing the computer

# upgrading the brain to Gradient Boosting 
# this is way better than random forest because it acts like a smart team that passes mistakes down the line and learns from them instead of just taking random votes
# it basically builds one small decision tree, looks at what it got wrong, and builds the next tree specifically to fix those mistakes
from sklearn.ensemble import GradientBoostingClassifier 

# grabbing this balancing tool to mathematically punish the AI if it gets biased and just wants to buy all the time
# because the market goes up more often than down over long periods, ai naturally gets lazy and just guesses "buy" to get a good score. this tool forces it to work hard for "sell" signals too.
from sklearn.utils.class_weight import compute_sample_weight


# --- section 2: configuration (exness connection) ---
SYMBOL = "EURUSDm"

# use these for your demo account first
MY_LOGIN = 298625031         
MY_PASSWORD = "Maybe14031997&"  
MY_SERVER = "Exness-MT5Trial9"

# trading the 4h chart to filter out all that random noise you get on the 15-minute charts
# big banks and institutions trade the 4h and daily charts, so we want to ride their waves instead of fighting the chop
TIMEFRAME = mt5.TIMEFRAME_H4 

# bumped this up to 70% meaning the AI has to be screaming with confidence before we risk money
# we don't want the bot gambling on 51% guesses. 0.70 means it sees a very clear historical pattern
CONFIDENCE_THRESHOLD = 0.70 

# choppy market filter using adx... anything under 22 means the market is dead and moving sideways so we stay out
# adx doesn't tell us if it's going up or down, it just tells us if there is enough momentum to actually hit our take profit
ADX_THRESHOLD = 22 


# --- section 3: the trading hands ---
def check_open_positions(symbol):
    # quick safety check before doing anything
    # if a trade is already running just hold fire because we only want one trade open at a time to manage risk
    # this stops the bot from opening 50 trades in a row and blowing your account if it gets confused
    positions = mt5.positions_get(symbol=symbol)
    
    if positions is None or len(positions) == 0:
        return False # coast is clear, no trades open
    
    return True # already in a trade so just wait for it to hit stop loss or take profit

def send_trade_order(symbol, type_trade, price, sl, tp):
    # packaging the exact order details to send over to the mt5 broker
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01, # trading exactly one micro lot so we only risk pennies while testing
        "type": type_trade, # tells the broker if it is a buy or a sell
        "price": price, # the exact ask or bid price right this second
        "sl": sl, # where we cut our losses
        "tp": tp, # where we take our money and run
        "magic": 234000, # tracking id so we know exactly which trades came from this specific bot
        "comment": "Smart AI H4 Bot",
        "type_time": mt5.ORDER_TIME_GTC, # good till cancelled
        "type_filling": mt5.ORDER_FILLING_FOK, # fill or kill meaning give me the exact price right now or cancel it entirely
    }
    
    # sending the package to exness
    result = mt5.order_send(request)
    
    # failsafe in case the market is moving too fast and rejects the FOK order
    # just immediately try again with IOC (immediate or cancel) to make sure we get into the trade before it leaves without us
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        request["type_filling"] = mt5.ORDER_FILLING_IOC
        result = mt5.order_send(request)
        
    return result


# --- section 4: the brain (data processing) ---
def resample_to_4h(df_1h):
    # neat trick to build our own super clean 4h candles directly from 1h broker data
    # we do this because sometimes asking the broker directly for 4h data gives us weird time zones
    aggregation = {
        'Open': 'first', # the opening price of the first hour
        'High': 'max',   # the highest price reached during the 4 hours
        'Low': 'min',    # the lowest price reached during the 4 hours
        'Close': 'last', # the closing price of the final hour
        'Volume': 'sum'  # total activity combined
    }
    
    # pandas doing the heavy math to group 4 individual hours into one solid block
    df_4h = df_1h.resample('4h').agg(aggregation)
    
    # drop the very last row since that 4h candle is still moving and hasn't closed yet
    # we never want the ai making decisions on a candle that is still jumping around
    df_4h.dropna(inplace=True)
    return df_4h

def prepare_data(df):
    # feeding the ai the clues it needs to understand the market breathing
    # raw prices aren't enough, it needs context to see the invisible trends
    
    # adding moving averages for the macro and micro trend direction
    df["SMA_50"] = ta.SMA(df["Close"], timeperiod=50) # fast trend
    df["SMA_200"] = ta.SMA(df["Close"], timeperiod=100) # the heavy trend boss to veto bad trades
    
    # rsi to see if we are overbought or oversold (like a rubber band stretched too far)
    df["RSI"] = ta.RSI(df["Close"], timeperiod=14)
    
    # atr to measure the wildness and volatility so we can set dynamic stop losses later
    # if atr is high, the market is violent, so we need wider stops to survive
    df["ATR"] = ta.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
    
    # adx to check trend strength (are we actually moving somewhere or just pacing back and forth?)
    df["ADX"] = ta.ADX(df["High"], df["Low"], df["Close"], timeperiod=14)
    
    # pro trick here adding lag features to show momentum
    # showing the ai what the rsi was one candle ago is way better because it sees the actual movement not just a flat number
    df["RSI_Lag1"] = df["RSI"].shift(1)
    df["Close_Lag1"] = df["Close"].shift(1)
    
    # clean up the blank spaces created by indicator math at the beginning of the timeline
    df.dropna(inplace=True)
    return df


# --- section 5: the strategy loop ---

# i am telling the bot to log in specifically to my exness account
# this throws the physical switch to connect your code to your money
if not mt5.initialize(login=MY_LOGIN, password=MY_PASSWORD, server=MY_SERVER):
    print(f"!!! LOGIN FAILED !!! Error: {mt5.last_error()}")
    mt5.shutdown()
    quit() 
else:
    print(f">>> SUCCESS: Logged into Exness Account {MY_LOGIN}")
    # the fix: python is too fast! we must force it to wait 3 seconds here.
    # this gives the mt5 desktop app time to fully sync with the exness server before we ask for data.
    time.sleep(3)
    
print(f"Morning Mr Mazivanhanga ::: System Online :: {SYMBOL} H4 Strategy Active.")
print("Waiting for top of the hour (Minute :00)...")

# giving the bot a memory starting at -1
# this ensures the heavy brain training only happens once a day so the hourly loop stays lightning fast
last_trained_day = -1 
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)


# --- the mock exam (testing the robot before live trading) ---
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
print("--- RUNNING AI MOCK EXAM ---")

# the fix: we already logged in perfectly at the top of section 5! 
# asking mt5 to log in *again* right here causes a double-knock, making it panic and throw the -1 error.
# i have deleted the second mt5.initialize() command entirely.

# smart check: making absolutely sure exness isn't hiding a small letter at the end of eurusd
symbol_info = mt5.symbol_info(SYMBOL)

if symbol_info is None:
    print(f"!!! error: exness says '{SYMBOL}' does not exist on this specific account.")
    # asking exness what it actually calls eurusd behind the scenes
    all_symbols = mt5.symbols_get()
    eurusd_symbols = [s.name for s in all_symbols if "EURUSD" in s.name]
    print(f"!!! please go to section 2 and change SYMBOL = '{SYMBOL}' to one of these exact names: {eurusd_symbols}")
    quit()

# forcing the mt5 terminal to physically select the correct symbol in the market watch
if not mt5.symbol_select(SYMBOL, True):
    print(f"failed to wake up {SYMBOL} in mt5. error: {mt5.last_error()}")
    quit()
# asking for 10,000 hourly candles to study
rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 10000)

if rates is None or len(rates) == 0:
    # adding the exact mt5 error code so we stop guessing if it fails
    print(f"!!! MT5 INTERNAL ERROR CODE: {mt5.last_error()}")
    print("!!! error: the broker didn't send the data. please open a EURUSD H1 chart in MT5 and scroll left to download history.")
    quit()

# converting the raw numbers into our invisible spreadsheet
data = pd.DataFrame(rates)

# ensuring the time column is handled correctly regardless of broker version
# some brokers call it 'time', some call it 'Date'. this code forces it to be 'Date' so our math works.
if 'time' in data.columns:
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
elif 'Date' not in data.columns:
    print("!!! error: 'time' column not found in mock exam data. check broker feed.")
    quit()
    
# making time the backbone of our spreadsheet
data.set_index('Date', inplace=True)

# format to 4h and add the technical indicators
df_4h = resample_to_4h(data)
df = prepare_data(df_4h)

# building the answer key
# if the next close is higher than the current close then it's a 1 (buy) otherwise 0 (sell)
df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

# setting up the study material and the answers
features = ["SMA_50", "SMA_200", "RSI", "RSI_Lag1", "ADX", "ATR"]
X = df[features] # the clues
y = df["Target"] # the answers

# chopping the timeline 80% for studying and 20% for the hidden exam
# shuffle is false because time order matters and we cant mix up the days like a deck of cards
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

print("1. AI is studying the past data...")

# calculating class weights so the bot gets punished if it just lazily guesses buy all the time
exam_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# creating the actual brain structure
test_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# forcing the bot to study with the balancing weights attached
test_model.fit(X_train, y_train, sample_weight=exam_weights)

print("2. AI is taking the strict final exam using our custom rules...")

# instead of asking for a raw guess we ask the ai for the exact percentages using predict_proba
exam_probabilities = test_model.predict_proba(X_test)

# we need empty lists to hold our strict filtered trades
strict_predictions = []
strict_y_test = [] # keeping track of the actual answers only for the trades we actually took

# looping through every single candle in the hidden exam
for i in range(len(X_test)):
    prob_down = exam_probabilities[i][0] # confidence for a sell
    prob_up = exam_probabilities[i][1] # confidence for a buy
    
    # pulling the price and the trend boss line for this specific past candle
    # since we are doing a mock exam we have to simulate looking at the chart visually
    price = df.iloc[len(X_train) + i]['Close'] 
    sma_200 = X_test.iloc[i]['SMA_200']
    actual_target = y_test.iloc[i]
    
    # simulating the buy logic
    if prob_up > CONFIDENCE_THRESHOLD:
        if price > sma_200:
            strict_predictions.append(1) # trade approved by both ai and the trend boss
            strict_y_test.append(actual_target)
            
    # simulating the sell logic
    elif prob_down > CONFIDENCE_THRESHOLD:
        if price < sma_200:
            strict_predictions.append(0) # trade approved
            strict_y_test.append(actual_target)

# checking if the bot actually took any trades under these strict rules
if len(strict_predictions) == 0:
    print("\n* Wow! The rules were so strict the bot refused to trade entirely.")
    print("* This is actually better than losing money in a bad market.")
else:
    # 5. the new report card (only grading the trades we actually took)
    acc = accuracy_score(strict_y_test, strict_predictions)
    print(f"\n* Strict Overall Accuracy: {acc * 100:.2f}%")

    # precision is the most important one now
    # precision means: out of all the times the bot said "trade", how many were actually winners?
    prec = precision_score(strict_y_test, strict_predictions, zero_division=0)
    print(f"* Strict Trading Precision: {prec * 100:.2f}%")

    print(f"\n* Total Trades Taken: {len(strict_predictions)} out of {len(X_test)} possible candles")
    
    print("\n* Strict Confusion Matrix (Scorecard):")
    print(confusion_matrix(strict_y_test, strict_predictions))
    print("----------------------------\n")


# --- live trading loop starts here ---
while True:
    now = dt.datetime.now()
    
    # check if its a weekday because forex sleeps on weekends
    if now.weekday() < 5: 
        
        # waiting right at the start of a new hour 
        # doing it between minute 0 and 2 gives the broker time to finalize the previous candle
        if 0 <= now.minute <= 2:
            
            # i removed the duplicate naked mt5.initialize() here too to prevent connection drops
            mt5.initialize(login=MY_LOGIN, password=MY_PASSWORD, server=MY_SERVER)
            
            # connect to broker and see if we are already in a trade
            if check_open_positions(SYMBOL):
                print(f"[{now.hour}:{now.minute}] Trade is LIVE. Managing position... (No new entry)")
                mt5.shutdown() # disconnect temporarily to save computer memory
                time.sleep(300) # sleep for 5 minutes and check again
                continue
            
            print(f"\n--- ANALYZING 4H MARKET at {now.hour}:{now.minute} ---")
            
            # pulling the freshest data direct from server for perfect accuracy
            rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 1500)
            
            if rates is None or len(rates) == 0:
                print("Error: No data fetched from MT5. Retrying next hour.")
                time.sleep(65)
                continue

            # turning raw computer arrays into a readable dataframe table
            data = pd.DataFrame(rates)
            
            # ensuring the live data also uses the correct 'Date' index
            # making sure live data matches our mock exam data perfectly
            if 'time' in data.columns:
                data['time'] = pd.to_datetime(data['time'], unit='s')
                data.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
                data.set_index('Date', inplace=True)
            elif 'Date' not in data.columns:
                print("Error: 'time' column not found in Live Market data.")
                continue

            # squish to 4h and add clues
            df_4h = resample_to_4h(data)
            df = prepare_data(df_4h)
            
            # target column for the daily retrain
            df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
            
            features = ["SMA_50", "SMA_200", "RSI", "RSI_Lag1", "ADX", "ATR"]
            
            # daily memory check to see if we need to do a heavy workout today
            # we only want to retrain the brain once a day so the hourly checks happen instantly
            if now.day != last_trained_day:
                print(">>> 🧠 Waking up the AI: Training the Brain on fresh data...")
                
                # drop the last unfinished candle so we only train on closed history
                train_df = df.iloc[:-1]
                X = train_df[features]
                y = train_df["Target"]
                
                # applying the same balancing trick to the live brain
                live_weights = compute_sample_weight(class_weight='balanced', y=y)
                
                # building the real live brain
                model.fit(X, y, sample_weight=live_weights) 
                
                # updating memory so we skip this heavy math until tomorrow
                last_trained_day = now.day 
                print(">>> ✅ Brain Trained! Ready for lightning-fast trading.")
            else:
                print(">>> ⚡ Brain already trained today. Skipping straight to live prediction.")
            
            # grabbing the very last candle to predict what happens right now
            current_data = df.iloc[[-1]][features]
            
            # getting the exact percentage of confidence instead of just a raw guess
            probability = model.predict_proba(current_data)[0]
            
            current_adx = current_data["ADX"].values[0]
            current_atr = current_data["ATR"].values[0]
            current_price = df.iloc[-1]["Close"]
            
            # pulling the trend boss line to make sure we don't buy in a crash
            current_sma_200 = current_data["SMA_200"].values[0]
            
            print(f"Current ADX (4H): {current_adx:.2f} (Trend Strength)")
            print(f"Bot_Mazi Confidence: SELL {probability[0]*100:.1f}% | BUY {probability[1]*100:.1f}%")

            # execution logic and filters
            
            # checking if the market is actually moving or just chopping sideways
            if current_adx > ADX_THRESHOLD:
                
                # making sure the bot is super confident before moving forward
                if probability[1] > CONFIDENCE_THRESHOLD or probability[0] > CONFIDENCE_THRESHOLD:
                    
                    # split second tick check to prevent crashing if the broker connection drops mid-thought
                    tick = mt5.symbol_info_tick(SYMBOL)
                    
                    if tick is None:
                        print(f"CRITICAL WARNING: Cannot get live price for {SYMBOL}. Skipping trade to avoid crash.")
                    else:
                        
                        # logic for going long (buying)
                        if probability[1] > CONFIDENCE_THRESHOLD:
                            # trend boss veto rule 
                            # physically blocking the trade if trying to buy under the 200 moving average
                            if current_price > current_sma_200:
                                print(">>> SIGNAL: STRONG BUY (High Confidence + Macro Uptrend)")
                                
                                # setting dynamic stops based on current volatility instead of rigid pips
                                # if the market is crazy today, atr gives us a wider breathing room
                                sl = current_price - (current_atr * 1.5)
                                tp = current_price + (current_atr * 2.5) 
                                
                                send_trade_order(SYMBOL, mt5.ORDER_TYPE_BUY, tick.ask, sl, tp)
                                time.sleep(300) # sleep for 5 minutes after firing
                            else:
                                print(">>> VETO: AI wants to Buy, but price is BELOW the 200 SMA. Trade Cancelled!")
                            
                        # logic for shorting (selling)
                        elif probability[0] > CONFIDENCE_THRESHOLD:
                            # trend boss veto rule for sells
                            if current_price < current_sma_200:
                                print(">>> SIGNAL: STRONG SELL (High Confidence + Macro Downtrend)")
                                
                                sl = current_price + (current_atr * 1.5)
                                tp = current_price - (current_atr * 2.5)
                                
                                send_trade_order(SYMBOL, mt5.ORDER_TYPE_SELL, tick.bid, sl, tp)
                                time.sleep(300)
                            else:
                                print(">>> VETO: AI wants to Sell, but price is ABOVE the 200 SMA. Trade Cancelled!")
                
                else:
                    # just sitting on hands if confidence is hovering around 50/50
                    print(">>> Mr Mazi The SIGNAL is weak (retracement): WEAK / UNCERTAIN. Therefore Dont Trade.")
            
            else:
                # low adx warning
                print(">>> MARKET CHOPPY (Low ADX). Staying safe (No Trade).")

            # cutting the connection to keep memory usage low while waiting for the next hour
            mt5.shutdown()
            print("Analysis Complete. Sleeping...")
            
            # forcing a full minute sleep so the loop doesn't accidentally run twice in the same minute window
            time.sleep(60)
            
    # the main heartbeat just waking up every half minute to check the clock
    time.sleep(30)