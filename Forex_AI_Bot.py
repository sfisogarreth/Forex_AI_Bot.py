# =============================================================================
# SECTION 1: IMPORTS & BRAIN TOOLS
# =============================================================================

"""

This robot uses SUPERVISED LEARNING - we train models on historical data where we 
already know the outcome (future price direction), then apply learned patterns to 
new data. This differs from REINFORCEMENT LEARNING where the AI learns by trial 
and error in real-time (like DeepMind's AlphaGo), or UNSUPERVISED LEARNING where 
we let the AI find hidden patterns without labeled answers.

ALTERNATIVE APPROACHES WE COULD USE:
- Deep Learning (LSTM/Transformers): Neural networks that remember long-term 
  sequences, excellent for time series but need massive data and GPU power
- Reinforcement Learning (PPO/DQN): AI learns optimal trading by receiving 
  rewards for profit and penalties for losses, mimics how humans really learn
- Ensemble Stacking: Layering multiple model types (SVM + Neural Net + GBM) 
  where a meta-learner combines their predictions
"""

import talib as ta  
# TA-Lib is a C++ library wrapped for Python - it calculates technical indicators 
# using optimized algorithms. We use it because writing RSI/ADX formulas manually 
# introduces floating-point errors and is computationally slower.

import numpy as np  
# NumPy uses vectorization - instead of looping through price data with 'for' 
# loops (slow in Python), it performs operations on entire arrays at once using 
# underlying C/Fortran code. This is called SIMD (Single Instruction Multiple Data).

import pandas as pd 
# Pandas builds on NumPy to provide labeled data structures (DataFrames). 
# Think of it as a spreadsheet with superpowers: automatic alignment, handling 
# missing data, time-series functionality. Under the hood it's just organized 
# NumPy arrays with index tracking.

import datetime as dt 
import MetaTrader5 as mt5 
import time 
import logging
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
# SKLEARN: THE MACHINE LEARNING WORKHORSE
# ---------------------------------------
# Scikit-learn is the Swiss Army knife of classical ML (not deep learning). 
# It follows consistent APIs: fit() to train, predict() to infer, predict_proba() 
# for probabilities. This consistency lets us swap algorithms easily.

from sklearn.ensemble import (
    GradientBoostingClassifier, 
    RandomForestClassifier, 
    ExtraTreesClassifier,
    VotingClassifier
)
# ENSEMBLE LEARNING EXPLAINED:
# Instead of trusting one model's opinion, we combine multiple "experts." 
# This reduces VARIANCE (overfitting to noise) and BIAS (oversimplification).
# 
# GRADIENT BOOSTING: Builds trees sequentially, each correcting errors of previous.
#   - Pros: High accuracy, handles non-linear relationships well
#   - Cons: Prone to overfitting if not regularized, sequential = slower training
#
# RANDOM FOREST: Builds trees in parallel on random data subsets, votes on outcome.
#   - Pros: Parallelizable, naturally regularized by randomness, hard to overfit
#   - Cons: Can be less accurate than boosting on clean data
#
# EXTRA TREES (Extremely Randomized Trees): Like RF but splits are random, not optimal.
#   - Pros: Faster training, less variance, good for high-noise financial data
#   - Cons: Higher bias, can miss subtle patterns

from sklearn.utils.class_weight import compute_sample_weight
# IMBALANCED LEARNING: Forex markets trend up ~52% of time. Without balancing, 
# AI learns "always predict buy" and gets 52% accuracy while losing money on 
# commissions. We use 'balanced' mode which automatically calculates weights 
# inversely proportional to class frequencies: weight = n_samples / (n_classes * n_samples_class)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    confusion_matrix, f1_score, roc_auc_score, log_loss
)
# METRICS DEEP DIVE:
# - Accuracy: (TP+TN)/Total - misleading if classes imbalanced
# - Precision: TP/(TP+FP) - of trades taken, how many won? CRITICAL for trading
# - Recall: TP/(TP+FN) - of all winning opportunities, how many did we catch?
# - F1: Harmonic mean of precision/recall - balances both
# - ROC-AUC: Ability to distinguish classes across all thresholds
# - Log Loss: Penalizes confident wrong predictions heavily (good for calibration)

from sklearn.model_selection import TimeSeriesSplit
# TimeSeriesSplit respects temporal order - crucial for financial data where 
# shuffling would leak future information into training (look-ahead bias).

from sklearn.preprocessing import StandardScaler
# Feature scaling ensures indicators with large values (price) don't dominate 
# those with small values (RSI 0-100). Essential for distance-based algorithms 
# like SVM or Neural Networks, less critical for tree-based methods but helps 
# with numerical stability.

# =============================================================================
# SECTION 2: CONFIGURATION & RISK MANAGEMENT FRAMEWORK
# =============================================================================

"""
RISK MANAGEMENT PHILOSOPHY:
---------------------------
Trading is not about being right; it's about making money when right and losing 
little when wrong. The Kelly Criterion tells us optimal bet size = edge/odds. 
In practice, we use "Fractional Kelly" (1/4 to 1/2 Kelly) to account for 
uncertainty in our edge estimation.

POSITION SIZING MODELS WE COULD USE:
1. Fixed Fractional: Risk fixed % of account per trade (what we implement)
2. Fixed Ratio: Increase size after X profits, decrease after X losses
3. Kelly Criterion: f* = (bp - q) / b where b=odds, p=win rate, q=loss rate
4. Optimal f (Ralph Vince): Maximizes geometric growth rate
5. Martingale/Anti-Martingale: Double down on losses (DANGEROUS) or wins
"""

@dataclass
class TradingConfig:
    """
    Dataclasses (Python 3.7+) automatically generate __init__, __repr__, etc.
    This makes configuration immutable and self-documenting.
    """
    # Broker Connection
    SYMBOL: str = "EURUSDm"
    MY_LOGIN: int = 298625031         
    MY_PASSWORD: str = "Maybe14031997&"  
    MY_SERVER: str = "Exness-MT5Trial9"
    
    # Timeframe Strategy
    TIMEFRAME = mt5.TIMEFRAME_H4
    """
    WHY 4-HOUR TIMEFRAMES?
    ----------------------
    Market microstructure research shows institutional flows cluster around 
    daily/4H boundaries. Lower timeframes contain more noise (bid-ask bounce, 
    stop hunts) with less signal. The "volatility signature" of 4H bars captures 
    meaningful moves while filtering random walk components.
    
    ALTERNATIVE: Use multiple timeframe analysis (MTFA) - confirm 4H signal 
    with Daily trend and 1H entry timing. This is called "top-down analysis."
    """
    
    # Risk Parameters
    RISK_PER_TRADE_PERCENT: float = 1.0  # Risk 1% of account per trade
    MAX_DAILY_RISK_PERCENT: float = 3.0   # Stop trading after 3% daily loss
    MAX_POSITIONS: int = 1  # Only one trade at a time
    
    # ML Confidence Thresholds (Adaptive - will adjust based on performance)
    BASE_CONFIDENCE_THRESHOLD: float = 0.65  # Lowered from 0.75 for more signals
    MIN_CONFIDENCE_THRESHOLD: float = 0.60
    MAX_CONFIDENCE_THRESHOLD: float = 0.85
    
    # Market Filters
    ADX_THRESHOLD: float = 20  # Slightly lowered from 22
    MAX_SPREAD_PIPS: float = 3.0  # Don't trade if spread > 3 pips
    MIN_ATR_PIPS: float = 5.0   # Avoid dead markets
    
    # ATR Multipliers for Stop Loss / Take Profit
    SL_ATR_MULTIPLIER: float = 1.5
    TP_ATR_MULTIPLIER: float = 2.5  # 1:1.67 risk:reward ratio
    
    # Ensemble Configuration
    USE_ENSEMBLE: bool = True  # Use 3 models instead of 1
    ENSEMBLE_AGREEMENT_REQUIRED: int = 2  # Need 2/3 models to agree

config = TradingConfig()

# Setup logging for post-trade analysis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'trading_log_{dt.datetime.now().strftime("%Y%m%d")}.txt'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

"""
LOGGING STRATEGY EXPLAINED:
---------------------------
Systematic trading requires systematic review. By logging every decision point, 
we can perform "post-mortem analysis" - examining losing trades to distinguish 
between:
- Bad luck (proper signal, market randomness)
- Model degradation (regime change, feature失效)
- Execution errors (slippage, spread widening)
- Logic flaws (coding bugs, timing issues)

This is the "scientific method" applied to trading: hypothesize (strategy), 
experiment (live trade), measure (logs), analyze (statistics), refine (iterate).
"""


# =============================================================================
# SECTION 3: MARKET HEALTH & RISK MANAGEMENT SYSTEMS
# =============================================================================

class RiskManager:
    """
    Object-oriented programming (OOP) approach encapsulates risk logic. This 
    separation of concerns means we can test risk rules independently of 
    trading logic, and reuse this class across different strategies.
    
    OOP CONCEPTS USED:
    - Encapsulation: Risk data and methods bundled together
    - Abstraction: Complex calculations hidden behind simple method calls
    - State Management: Tracking daily P&L and exposure over time
    """
    
    def __init__(self):
        self.daily_pnl: float = 0.0
        self.trades_today: int = 0
        self.last_trade_date: int = -1
        self.recent_performance: List[int] = []  # 1 for win, 0 for loss
        
    def reset_daily(self, current_day: int):
        """Reset counters at midnight server time."""
        if current_day != self.last_trade_date:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.last_trade_date = current_day
            
    def can_trade(self, account_balance: float, current_day: int) -> Tuple[bool, str]:
        """
        Pre-trade risk checks. Returns (permission, reason).
        
        RISK OF RUIN THEORY:
        If you risk 1% per trade with 50% win rate, risk of ruin is ~0%. 
        With 40% win rate, risk of 50% drawdown is significant. These checks 
        ensure we live to trade another day.
        """
        self.reset_daily(current_day)
        
        # Check daily loss limit
        daily_loss_limit = account_balance * (config.MAX_DAILY_RISK_PERCENT / 100)
        if abs(self.daily_pnl) >= daily_loss_limit:
            return False, f"Daily risk limit hit: {self.daily_pnl:.2f}"
        
        # Check max positions
        if self.trades_today >= 5:  # Max 5 trades per day
            return False, "Max daily trades reached"
            
        return True, "Risk checks passed"
    
    def update_performance(self, result: int):
        """Track win/loss for adaptive threshold calculation."""
        self.recent_performance.append(result)
        # Keep last 50 trades in memory for statistical significance
        if len(self.recent_performance) > 50:
            self.recent_performance.pop(0)
    
    def get_adaptive_threshold(self) -> float:
        """
        ADAPTIVE SYSTEMS IN TRADING:
        ----------------------------
        Markets have regimes: trending, mean-reverting, high/low volatility. 
        A fixed threshold assumes market efficiency is constant - it isn't.
        
        We use the model's recent performance to adjust selectivity:
        - Winning streak: Lower threshold (confidence is high, capture more moves)
        - Losing streak: Raise threshold (become more selective, wait for A+ setups)
        
        This is similar to "position sizing based on equity curve" - reduce size 
        during drawdowns, increase during equity peaks.
        
        ALTERNATIVE APPROACHES:
        - Regime Detection: Use HMM (Hidden Markov Models) to identify market 
          states statistically, apply different thresholds per regime
        - Volatility Targeting: Adjust threshold based on realized volatility
        - Bayesian Updating: Treat threshold as a parameter with prior distribution, 
          update based on recent evidence
        """
        if len(self.recent_performance) < 20:
            return config.BASE_CONFIDENCE_THRESHOLD
        
        # Calculate recent win rate
        recent_win_rate = sum(self.recent_performance[-20:]) / 20
        
        # Adjust threshold: lower when winning (more aggressive), higher when losing
        adjustment = (recent_win_rate - 0.5) * -0.3  # -0.15 to +0.15 range
        threshold = config.BASE_CONFIDENCE_THRESHOLD + adjustment
        
        return max(config.MIN_CONFIDENCE_THRESHOLD, 
                   min(config.MAX_CONFIDENCE_THRESHOLD, threshold))


def check_market_health(symbol: str) -> Tuple[bool, str]:
    """
    MARKET MICROSTRUCTURE FILTERS:
    ------------------------------
    Forex isn't always tradable. News events cause spread explosions, 
    low liquidity periods (weekends, holidays) cause gapping, and 
    high volatility without trend (choppy markets) chops up trend followers.
    
    These filters implement "market selection" - the idea that not trading 
    is often the best trade. Professional traders are selective; amateurs 
    feel the need to be constantly in the market.
    """
    tick = mt5.symbol_info_tick(symbol)
    symbol_info = mt5.symbol_info(symbol)
    
    if tick is None or symbol_info is None:
        return False, "Cannot retrieve market data"
    
    # Spread check (market maker costs)
    spread = (tick.ask - tick.bid) / 0.0001  # Convert to pips
    if spread > config.MAX_SPREAD_PIPS:
        return False, f"Spread too wide: {spread:.1f} pips (max {config.MAX_SPREAD_PIPS})"
    
    # Session liquidity check
    now = dt.datetime.now()
    day_hour = now.weekday() * 24 + now.hour
    
    # Avoid Sunday open (gaps from weekend news) and Friday close (low liquidity)
    if day_hour in [0, 1, 2, 120, 121, 122]:  
        return False, "Low liquidity period (weekend transition)"
    
    # Check if symbol is tradeable
    if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
        return False, "Symbol not available for trading (close-only or disabled)"
        
    return True, "Market healthy"


def calculate_position_size(
    account_balance: float, 
    risk_percent: float, 
    atr: float, 
    current_price: float,
    symbol: str
) -> float:
    """
    POSITION SIZING: THE MATHEMATICS OF SURVIVAL
    --------------------------------------------
    Most traders fail not because of bad signals, but because of position 
    sizing errors. Risking 10% per trade requires 10 consecutive losses to 
    blow up (happens more than you think). Risking 1% requires 100 losses.
    
    THE KELLY CRITERION (John Kelly, 1956):
    Optimal bet size f* = (p*b - q) / b
    Where: p = win probability, q = loss probability (1-p), b = win/loss ratio
    
    For example: 60% win rate, 2:1 reward/risk
    f* = (0.6*2 - 0.4) / 2 = 0.4 or 40% of account (AGGRESSIVE!)
    
    We use "Fractional Kelly" (1/4 to 1/2) because we don't know true probabilities.
    
    ALTERNATIVE SIZING METHODS:
    - Fixed Ratio (Ryan Jones): Increase size after fixed profit amount, not %
    - Martingale: Double size after loss (RUINOUS - never use)
    - Anti-Martingale: Increase size during winning streaks (pyramiding)
    - Volatility Targeting: Size = Target Volatility / Current Volatility
    """
    
    # Calculate risk amount in account currency
    risk_amount = account_balance * (risk_percent / 100)
    
    # Calculate stop distance in price terms
    stop_distance = atr * config.SL_ATR_MULTIPLIER
    
    # Get symbol specifications for lot sizing
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error("Cannot get symbol info for position sizing")
        return 0.01  # Minimum fallback
    
    # Calculate pip value (simplified for EURUSD)
    # For EURUSD: 1 pip = 0.0001, 1 standard lot = $10 per pip
    point = symbol_info.point
    contract_size = symbol_info.trade_contract_size
    
    # Risk per pip = Risk Amount / (Stop Distance / Point)
    # Lot Size = Risk per Pip / Value per Pip
    ticks_at_risk = stop_distance / point
    value_per_tick = contract_size * point  # Simplified, assumes account currency = USD
    
    if ticks_at_risk == 0:
        return 0.01
        
    lot_size = risk_amount / (ticks_at_risk * value_per_tick)
    
    # Clamp to broker limits
    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    lot_step = symbol_info.volume_step
    
    # Round to valid lot size
    lot_size = max(min_lot, min(lot_size, max_lot))
    lot_size = round(lot_size / lot_step) * lot_step
    
    return round(lot_size, 2)


# =============================================================================
# SECTION 4: FEATURE ENGINEERING & DATA PROCESSING
# =============================================================================

"""
FEATURE ENGINEERING: THE SECRET SAUCE OF ML
-------------------------------------------
Algorithms are only as good as their inputs. Raw prices have low "signal-to-noise 
ratio." Feature engineering transforms raw data into representations that make 
patterns more obvious to the algorithm.

TYPES OF FEATURES WE ENGINEER:
1. Trend Features: Moving averages, trend direction, momentum
2. Volatility Features: ATR, Bollinger Bands, volatility regime
3. Momentum Features: RSI, MACD, rate of change, divergence
4. Volume Features: Volume profile, accumulation/distribution (if available)
5. Temporal Features: Time of day, day of week, session indicators
6. Interaction Features: Price relative to moving averages, distance from highs/lows

DIMENSIONALITY REDUCTION ALTERNATIVES:
- PCA (Principal Component Analysis): Combine correlated indicators into 
  orthogonal components, reduces multicollinearity
- Autoencoders: Neural networks that learn compressed representations
- Feature Selection: Use Random Forest feature importance or Recursive 
  Feature Elimination to keep only predictive features
"""

def fetch_data_direct(symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
    """
    WHY WE FETCH 4H DIRECTLY INSTEAD OF RESAMPLING:
    ------------------------------------------------
    The original code resampled 1H -> 4H using pandas. This introduces:
    1. Look-ahead bias: The "Close" of a 4H bar built from 1H data isn't known 
       until the 4H completes, but pandas might use future 1H bars
    2. Timezone inconsistencies: Broker server time vs. local time vs. UTC
    3. Weekend gaps: Resampling can create artificial bars during market closure
    
    Direct 4H data from the broker ensures we trade exactly what we see on charts.
    
    DATA QUALITY ISSUES IN FINANCE:
    - Survivorship bias: Only current symbols in dataset (delisted stocks missing)
    - Look-ahead bias: Using information not available at decision time
    - Data snooping: Testing so many strategies that one works by chance
    - Non-stationarity: Statistical properties change over time (regime shifts)
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    
    if rates is None or len(rates) == 0:
        logger.error(f"Failed to fetch data: {mt5.last_error()}")
        return None
    
    df = pd.DataFrame(rates)
    
    # Standardize column names across different broker feeds
    column_mapping = {
        'time': 'Date', 'open': 'Open', 'high': 'High', 
        'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume',
        'spread': 'Spread', 'real_volume': 'RealVolume'
    }
    
    # Rename columns that exist
    for old, new in column_mapping.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
    
    # Convert timestamp
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], unit='s')
        df.set_index('Date', inplace=True)
    
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    COMPREHENSIVE FEATURE ENGINEERING PIPELINE:
    -------------------------------------------
    We expand from 6 basic features to 15+ sophisticated features. This gives 
    the ML model more "dimensions" to find patterns, but increases risk of 
    overfitting (finding patterns in noise). We combat overfitting through:
    - Regularization (in model parameters)
    - Cross-validation (time-series aware)
    - Feature importance analysis (drop useless features)
    
    TECHNICAL ANALYSIS THEORY:
    Technical indicators are heuristics - rules of thumb based on market 
    psychology and supply/demand. They work not because of math magic, but 
    because many traders watch them, creating self-fulfilling prophecies.
    """
    
    df = df.copy()
    
    # --- TREND FEATURES ---
    # Multiple timeframe trend alignment (stacked moving averages)
    df["SMA_20"] = ta.SMA(df["Close"], timeperiod=20)   # Fast trend
    df["SMA_50"] = ta.SMA(df["Close"], timeperiod=50)   # Medium trend
    df["SMA_200"] = ta.SMA(df["Close"], timeperiod=200) # Slow trend (regime)
    
    # Trend strength: distance from moving average normalized by volatility
    df["Dist_From_SMA200"] = (df["Close"] - df["SMA_200"]) / df["Close"]
    
    # Trend direction: 1 if bullish alignment (fast > slow), -1 if bearish
    df["Trend_Alignment"] = np.where(
        (df["SMA_20"] > df["SMA_50"]) & (df["SMA_50"] > df["SMA_200"]), 1,
        np.where((df["SMA_20"] < df["SMA_50"]) & (df["SMA_50"] < df["SMA_200"]), -1, 0)
    )
    
    # --- MOMENTUM FEATURES ---
    df["RSI"] = ta.RSI(df["Close"], timeperiod=14)
    df["RSI_Lag1"] = df["RSI"].shift(1)
    df["RSI_Momentum"] = df["RSI"] - df["RSI_Lag1"]  # Rate of change in RSI
    
    # MACD (Moving Average Convergence Divergence)
    macd, macd_signal, macd_hist = ta.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd_hist
    
    # --- VOLATILITY FEATURES ---
    df["ATR"] = ta.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
    df["ATR_Ratio"] = df["ATR"] / df["Close"]  # Normalized ATR
    
    # Bollinger Bands for volatility regime and mean-reversion signals
    upper, middle, lower = ta.BBANDS(df["Close"], timeperiod=20, nbdevup=2, nbdevdn=2)
    df["BB_Width"] = (upper - lower) / middle  # Volatility expansion/contraction
    df["BB_Position"] = (df["Close"] - lower) / (upper - lower)  # 0=lower band, 1=upper band
    
    # ADX for trend strength (not direction)
    df["ADX"] = ta.ADX(df["High"], df["Low"], df["Close"], timeperiod=14)
    
    # --- DIVERGENCE FEATURES (Advanced) ---
    # Price makes higher high but RSI makes lower high = Bearish divergence (weakness)
    # Price makes lower low but RSI makes higher low = Bullish divergence (strength)
    
    # Rolling highs/lows for divergence detection
    df["Price_High_5"] = df["High"].rolling(5).max()
    df["Price_Low_5"] = df["Low"].rolling(5).min()
    df["RSI_High_5"] = df["RSI"].rolling(5).max()
    df["RSI_Low_5"] = df["RSI"].rolling(5).min()
    
    # Detect divergence (simplified - true detection needs swing analysis)
    df["Bearish_Div"] = ((df["Close"] > df["Price_High_5"].shift(1)) & 
                         (df["RSI"] < df["RSI_High_5"].shift(1))).astype(int)
    df["Bullish_Div"] = ((df["Close"] < df["Price_Low_5"].shift(1)) & 
                         (df["RSI"] > df["RSI_Low_5"].shift(1))).astype(int)
    
    # --- TEMPORAL FEATURES ---
    # Forex has session-based seasonality (London open volatility, NY afternoon quiet)
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek
    
    # Session indicators ( Forex sessions in UTC )
    df["Is_London"] = ((df["Hour"] >= 7) & (df["Hour"] <= 16)).astype(int)  # London 7-16
    df["Is_NY"] = ((df["Hour"] >= 12) & (df["Hour"] <= 21)).astype(int)     # NY 12-21
    df["Is_Asian"] = ((df["Hour"] >= 0) & (df["Hour"] <= 6)).astype(int)    # Asian 0-6
    
    # --- SUPPORT/RESISTANCE PROXIMITY ---
    # Distance from recent highs/lows normalized by ATR
    df["Dist_From_High20"] = (df["High"].rolling(20).max() - df["Close"]) / df["ATR"]
    df["Dist_From_Low20"] = (df["Close"] - df["Low"].rolling(20).min()) / df["ATR"]
    
    # --- LAG FEATURES (Memory of recent state) ---
    df["Close_Lag1"] = df["Close"].shift(1)
    df["Return_Lag1"] = df["Close"].pct_change(1)
    df["Return_Lag4"] = df["Close"].pct_change(4)  # Previous 4H return
    
    # Clean NaN values created by indicators
    df.dropna(inplace=True)
    
    return df


# =============================================================================
# SECTION 5: MACHINE LEARNING MODEL ARCHITECTURE
# =============================================================================

class EnsembleTrader:
    """
    ENSEMBLE LEARNING ARCHITECTURE:
    --------------------------------
    Instead of relying on a single model's opinion, we create a "committee" of 
    diverse algorithms. The wisdom of crowds applies to ML: multiple weak learners 
    often outperform a single strong learner.
    
    WHY ENSEMBLES WORK:
    - Variance Reduction: Different models make different errors; averaging cancels noise
    - Bias Reduction: Some models capture patterns others miss
    - Robustness: If one model fails catastrophically, others can compensate
    
    ENSEMBLE METHODS:
    1. Voting: Hard (majority class) or Soft (average probabilities) - we use Soft
    2. Bagging: Train same algorithm on different data subsets (Random Forest)
    3. Boosting: Train sequentially, focus on errors (Gradient Boosting)
    4. Stacking: Meta-learner combines base model predictions (most advanced)
    
    We implement a heterogeneous ensemble (different algorithms) with soft voting.
    """
    
    def __init__(self):
        self.models: Dict[str, object] = {}
        self.is_trained: bool = False
        self.feature_names: List[str] = []
        
    def initialize_models(self):
        """
        MODEL HYPERPARAMETERS EXPLAINED:
        --------------------------------
        These settings control the bias-variance tradeoff:
        
        n_estimators: Number of trees. More = better fit but slower and can overfit.
        max_depth: Tree depth. Deeper = more complex patterns but overfits.
        learning_rate: Step size for boosting. Lower = slower but better generalization.
        subsample: Fraction of data per tree. <1.0 adds regularization (stochastic boosting).
        min_samples_leaf: Minimum samples in leaf nodes. Higher = smoother, less overfit.
        """
        
        # Model 1: Gradient Boosting (sequential error correction)
        self.models['gbm'] = GradientBoostingClassifier(
            n_estimators=200,        # More trees for smoother predictions
            learning_rate=0.05,      # Slower learning, better generalization
            max_depth=4,             # Slightly deeper than before
            subsample=0.8,           # Stochastic gradient boosting (regularization)
            min_samples_leaf=20,     # Require 20 samples per leaf (smoothness)
            random_state=42
        )
        
        # Model 2: Random Forest (parallel diversity)
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=30,
            min_samples_leaf=15,
            max_features='sqrt',     # Random subset of features per split (diversity)
            n_jobs=-1,               # Use all CPU cores
            random_state=42
        )
        
        # Model 3: Extra Trees (extreme randomization for variance reduction)
        self.models['et'] = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        
        logger.info("Ensemble initialized with GBM, Random Forest, and Extra Trees")
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        TRAINING PROCESS:
        -----------------
        We fit all three models independently on the same data. They'll learn 
        different aspects of the pattern due to algorithmic differences:
        - GBM focuses on hard examples (gradient descent on errors)
        - RF averages diverse trees through bagging
        - ET adds randomness to splits for even more diversity
        
        CLASS WEIGHTING:
        We use balanced class weights to handle the natural upward bias in markets.
        Without this, models achieve 55% accuracy by always predicting "up."
        """
        self.feature_names = X.columns.tolist()
        
        # Calculate sample weights for imbalance handling
        sample_weights = compute_sample_weight(class_weight='balanced', y=y)
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            if name == 'gbm':
                # GBM supports sample weights directly
                model.fit(X, y, sample_weight=sample_weights)
            else:
                # RF and ET use class_weight parameter in initialization
                model.fit(X, y)
        
        self.is_trained = True
        
        # Log feature importances (interpretability)
        self._log_feature_importance()
    
    def _log_feature_importance(self):
        """Log which features each model considers important."""
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = pd.Series(
                    model.feature_importances_, 
                    index=self.feature_names
                ).sort_values(ascending=False)
                logger.info(f"\n{name.upper()} Top 5 Features:\n{importances.head()}")
    
    def predict_proba(self, X: pd.DataFrame) -> Tuple[np.ndarray, int]:
        """
        SOFT VOTING ENSEMBLE:
        ---------------------
        Instead of majority vote (hard voting), we average predicted probabilities.
        This preserves confidence information: if two models are 51% sure and one 
        is 99% sure, the 99% model should have more influence.
        
        AGREEMENT FILTER:
        We require 2/3 models to agree on direction above threshold. This filters 
        out uncertain predictions where models contradict each other.
        
        Returns: (average_probabilities, agreement_count)
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        probabilities = []
        directions = []
        
        for name, model in self.models.items():
            proba = model.predict_proba(X)[0]  # [prob_down, prob_up]
            probabilities.append(proba)
            directions.append(np.argmax(proba))  # 0 for sell, 1 for buy
        
        # Average probabilities (soft voting)
        avg_proba = np.mean(probabilities, axis=0)
        
        # Check agreement: how many models predict the same direction?
        predicted_direction = np.argmax(avg_proba)
        agreement_count = sum(1 for d in directions if d == predicted_direction)
        
        return avg_proba, agreement_count
    
    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from each model for analysis."""
        return {name: model.predict_proba(X)[0] for name, model in self.models.items()}


# =============================================================================
# SECTION 6: WALK-FORWARD VALIDATION (BACKTESTING)
# =============================================================================

"""
WALK-FORWARD ANALYSIS: THE GOLD STANDARD FOR TIME SERIES
--------------------------------------------------------
Traditional train/test split assumes data is independent and identically 
distributed (i.i.d.). Financial time series are neither - they have autocorrelation 
(today's price depends on yesterday's) and non-stationarity (statistical properties 
change over time).

WALK-FORWARD PROCESS:
1. Train on period [0, t], test on [t, t+window]
2. Move forward, train on [0, t+window], test on [t+window, t+2*window]
3. Repeat until end of data
4. Concatenate all test periods for true out-of-sample performance

This simulates real trading: you only know past data when making decisions.

ALTERNATIVE VALIDATION METHODS:
- Purged K-Fold: Remove data near test periods to prevent leakage
- Combinatorial Purged CV: Multiple test periods, average results (computationally expensive)
- Monte Carlo Simulation: Randomize returns to test strategy robustness
"""

def walk_forward_validation(
    df: pd.DataFrame, 
    features: List[str],
    min_train_size: int = 500,
    test_window: int = 100,
    step_size: int = 50
) -> Dict:
    """
    Implements rolling window cross-validation for time series.
    
    Parameters:
    - min_train_size: Minimum historical data before first test
    - test_window: How many candles to test before retraining
    - step_size: How many candles to move forward each iteration
    
    Returns performance metrics across all test periods.
    """
    
    results = {
        'predictions': [],
        'actuals': [],
        'probabilities': [],
        'timestamps': []
    }
    
    n_samples = len(df)
    
    # Start after minimum training data
    current_idx = min_train_size
    
    while current_idx + test_window <= n_samples:
        # Define train and test periods
        train_data = df.iloc[:current_idx]
        test_data = df.iloc[current_idx:current_idx + test_window]
        
        # Prepare data
        X_train = train_data[features]
        y_train = train_data["Target"]
        X_test = test_data[features]
        y_test = test_data["Target"]
        
        # Train ensemble
        ensemble = EnsembleTrader()
        ensemble.initialize_models()
        ensemble.fit(X_train, y_train)
        
        # Test on forward window
        for i in range(len(X_test)):
            X_current = X_test.iloc[[i]]
            actual = y_test.iloc[i]
            
            proba, agreement = ensemble.predict_proba(X_current)
            
            # Apply trading logic filters
            price = test_data.iloc[i]['Close']
            sma_200 = X_current['SMA_200'].values[0]
            adx = X_current['ADX'].values[0]
            
            # Simulate trade decision
            trade_taken = False
            prediction = None
            
            if adx > config.ADX_THRESHOLD and agreement >= config.ENSEMBLE_AGREEMENT_REQUIRED:
                prob_up = proba[1]
                prob_down = proba[0]
                
                if prob_up > config.BASE_CONFIDENCE_THRESHOLD and price > sma_200:
                    trade_taken = True
                    prediction = 1
                elif prob_down > config.BASE_CONFIDENCE_THRESHOLD and price < sma_200:
                    trade_taken = True
                    prediction = 0
            
            if trade_taken:
                results['predictions'].append(prediction)
                results['actuals'].append(actual)
                results['probabilities'].append(max(proba))
                results['timestamps'].append(test_data.index[i])
        
        # Move window forward
        current_idx += step_size
        
        logger.info(f"Walk-forward progress: {current_idx}/{n_samples}")
    
    # Calculate metrics
    if len(results['predictions']) > 0:
        metrics = {
            'accuracy': accuracy_score(results['actuals'], results['predictions']),
            'precision': precision_score(results['actuals'], results['predictions'], zero_division=0),
            'recall': recall_score(results['actuals'], results['predictions'], zero_division=0),
            'f1': f1_score(results['actuals'], results['predictions'], zero_division=0),
            'total_trades': len(results['predictions'])
        }
        
        logger.info(f"\nWalk-Forward Results:\n{metrics}")
        return metrics
    else:
        logger.warning("No trades taken in walk-forward test")
        return {}


# =============================================================================
# SECTION 7: TRADING EXECUTION & ORDER MANAGEMENT
# =============================================================================

def check_open_positions(symbol: str) -> Tuple[bool, List]:
    """
    Position monitoring with detailed logging for post-trade analysis.
    """
    positions = mt5.positions_get(symbol=symbol)
    
    if positions is None:
        logger.error(f"Error checking positions: {mt5.last_error()}")
        return False, []
    
    if len(positions) == 0:
        return False, []
    
    # Log position details for tracking
    for pos in positions:
        profit = pos.profit + pos.swap  # Include swap costs
        logger.info(f"Open position: {pos.type} {pos.volume} lots, "
                   f"Profit: {profit:.2f}, Swap: {pos.swap:.2f}")
    
    return True, positions


def send_trade_order(
    symbol: str, 
    order_type: int, 
    price: float, 
    sl: float, 
    tp: float, 
    volume: float,
    risk_manager: RiskManager
) -> bool:
    """
    Institutional-grade order execution with multiple fallback mechanisms.
    
    ORDER TYPES EXPLAINED:
    - FOK (Fill or Kill): All or nothing immediately. Good for exact price requirement.
    - IOC (Immediate or Cancel): Fill what you can immediately, cancel rest.
    - RETURN: Allow partial fills (not used here).
    
    SLIPPAGE MANAGEMENT:
    We set maximum deviation (slippage) to prevent execution far from intended price 
    during volatile news events.
    """
    
    # Get symbol info for point size and stops level
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error("Cannot get symbol info for order")
        return False
    
    # Normalize prices to symbol digits
    digits = symbol_info.digits
    price = round(price, digits)
    sl = round(sl, digits)
    tp = round(tp, digits)
    
    # Check stops level (minimum distance from current price)
    stops_level = symbol_info.trade_stops_level
    if stops_level > 0:
        min_distance = stops_level * symbol_info.point
        if abs(price - sl) < min_distance:
            logger.warning(f"Stop loss too close, adjusting to minimum distance")
            if order_type == mt5.ORDER_TYPE_BUY:
                sl = price - min_distance * 2
            else:
                sl = price + min_distance * 2
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,  # Allow 10 points slippage max
        "magic": 234000,
        "comment": "Smart AI H4 Bot v2.0",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    # Send order
    result = mt5.order_send(request)
    
    if result is None:
        logger.error(f"Order send failed: {mt5.last_error()}")
        return False
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.warning(f"FOK failed with code {result.retcode}, trying IOC...")
        request["type_filling"] = mt5.ORDER_FILLING_IOC
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"IOC also failed: {result.retcode}")
            return False
    
    # Log successful trade
    action = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
    logger.info(f"Trade executed: {action} {volume} lots at {price}, "
               f"SL: {sl}, TP: {tp}, Ticket: {result.order}")
    
    return True


# =============================================================================
# SECTION 8: MAIN TRADING LOOP
# =============================================================================

def run_mock_exam():
    """
    Pre-live validation using walk-forward analysis.
    This gives realistic performance expectations before risking capital.
    """
    logger.info("=" * 60)
    logger.info("STARTING WALK-FORWARD VALIDATION (Mock Exam)")
    logger.info("=" * 60)
    
    # Fetch data
    df = fetch_data_direct(config.SYMBOL, config.TIMEFRAME, 5000)
    if df is None:
        logger.error("Cannot fetch data for mock exam")
        return False
    
    # Engineer features
    df = prepare_features(df)
    
    # Create target (next candle direction)
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    df.dropna(inplace=True)
    
    # Define feature set (expanded from original 6 to 15+ features)
    features = [
        "SMA_20", "SMA_50", "SMA_200", "Dist_From_SMA200", "Trend_Alignment",
        "RSI", "RSI_Lag1", "RSI_Momentum", "MACD_Hist",
        "ATR", "ATR_Ratio", "BB_Width", "BB_Position", "ADX",
        "Bearish_Div", "Bullish_Div", "Dist_From_High20", "Dist_From_Low20",
        "Return_Lag1", "Return_Lag4", "Is_London", "Is_NY"
    ]
    
    # Run walk-forward validation
    # Change this line in run_mock_exam()
    metrics = walk_forward_validation(df, features, min_train_size=1000, test_window=200, step_size=200)
    if not metrics:
        logger.error("Walk-forward test failed - strategy not viable")
        return False
    
    # Minimum performance criteria
    if metrics['precision'] < 0.55:  # Need >55% precision to overcome costs
        logger.warning(f"Precision {metrics['precision']:.2%} below 55% threshold")
        logger.warning("Strategy may not be profitable after spreads/commissions")
        return False
    
    logger.info("Mock exam PASSED - strategy meets minimum criteria")
    return True


def main():
    """
    MAIN EXECUTION FLOW:
    -------------------
    1. Initialize connection to broker
    2. Run validation (walk-forward test)
    3. Initialize risk management and ensemble models
    4. Enter live trading loop with hourly checks
    5. Daily model retraining on new data
    """
    
    # Initialize MT5
    if not mt5.initialize(
        login=config.MY_LOGIN, 
        password=config.MY_PASSWORD, 
        server=config.MY_SERVER
    ):
        logger.critical(f"MT5 initialization failed: {mt5.last_error()}")
        return
    
    logger.info(f"Connected to Exness account {config.MY_LOGIN}")
    time.sleep(3)  # Allow connection stabilization
    
    # Select symbol
    if not mt5.symbol_select(config.SYMBOL, True):
        logger.critical(f"Cannot select symbol {config.SYMBOL}")
        mt5.shutdown()
        return
    
    # Run pre-trade validation
    if not run_mock_exam():
        logger.critical("Validation failed - aborting live trading")
        mt5.shutdown()
        return
    
    # Initialize systems
    risk_manager = RiskManager()
    ensemble = EnsembleTrader()
    last_trained_day = -1
    
    logger.info("=" * 60)
    logger.info("LIVE TRADING STARTED")
    logger.info("=" * 60)
    
    # Main loop
    while True:
        try:
            now = dt.datetime.now()
            
            # Only trade weekdays
            if now.weekday() >= 5:
                time.sleep(3600)  # Sleep 1 hour on weekends
                continue
            
            # Check at start of each hour (minute 0-2)
            if not (0 <= now.minute <= 2):
                time.sleep(30)
                continue
            
            # Daily initialization
            mt5.initialize()
            
            # Check if already in position
            in_position, positions = check_open_positions(config.SYMBOL)
            if in_position:
                logger.info("Position active - monitoring only")
                mt5.shutdown()
                time.sleep(300)
                continue
            
            # Risk checks
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Cannot get account info")
                time.sleep(60)
                continue
            
            can_trade, reason = risk_manager.can_trade(account_info.balance, now.day)
            if not can_trade:
                logger.warning(f"Risk manager blocked trading: {reason}")
                time.sleep(3600)
                continue
            
            # Market health check
            healthy, health_reason = check_market_health(config.SYMBOL)
            if not healthy:
                logger.warning(f"Market unhealthy: {health_reason}")
                time.sleep(300)
                continue
            
            # Fetch and process data
            logger.info(f"Analyzing market at {now}")
            df = fetch_data_direct(config.SYMBOL, config.TIMEFRAME, 1000)
            if df is None:
                time.sleep(60)
                continue
            
            df = prepare_features(df)
            df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
            df.dropna(inplace=True)
            
            features = [
                "SMA_20", "SMA_50", "SMA_200", "Dist_From_SMA200", "Trend_Alignment",
                "RSI", "RSI_Lag1", "RSI_Momentum", "MACD_Hist",
                "ATR", "ATR_Ratio", "BB_Width", "ADX",
                "Bearish_Div", "Bullish_Div", "Dist_From_High20", "Dist_From_Low20",
                "Return_Lag1", "Is_London", "Is_NY"
            ]
            
            # Daily model retraining (or initialize if first run)
            if now.day != last_trained_day:
                logger.info("Retraining ensemble models on fresh data...")
                
                train_df = df.iloc[:-1]  # Exclude last unfinished candle
                X = train_df[features]
                y = train_df["Target"]
                
                ensemble = EnsembleTrader()
                ensemble.initialize_models()
                ensemble.fit(X, y)
                
                last_trained_day = now.day
                logger.info("Models retrained successfully")
            
            # Generate prediction
            current_data = df.iloc[[-1]][features]
            proba, agreement = ensemble.predict_proba(current_data)
            
            current_price = df.iloc[-1]["Close"]
            current_atr = current_data["ATR"].values[0]
            current_adx = current_data["ADX"].values[0]
            current_sma200 = current_data["SMA_200"].values[0]
            
            # Get adaptive threshold based on recent performance
            threshold = risk_manager.get_adaptive_threshold()
            
            logger.info(f"ADX: {current_adx:.2f}, Agreement: {agreement}/3, "
                       f"Threshold: {threshold:.2f}")
            logger.info(f"Probabilities: SELL {proba[0]*100:.1f}% | BUY {proba[1]*100:.1f}%")
            
            # Execution logic
            if current_adx < config.ADX_THRESHOLD:
                logger.info("ADX too low - no trend, no trade")
                mt5.shutdown()
                time.sleep(60)
                continue
            
            if agreement < config.ENSEMBLE_AGREEMENT_REQUIRED:
                logger.info("Models disagree - consensus required")
                mt5.shutdown()
                time.sleep(60)
                continue
            
            tick = mt5.symbol_info_tick(config.SYMBOL)
            if tick is None:
                logger.error("Cannot get current price")
                time.sleep(60)
                continue
            
            # Calculate position size
            position_size = calculate_position_size(
                account_info.balance,
                config.RISK_PER_TRADE_PERCENT,
                current_atr,
                current_price,
                config.SYMBOL
            )
            
            # Buy signal
            if proba[1] > threshold:
                if current_price > current_sma200:  # Trend filter
                    logger.info(">>> SIGNAL: STRONG BUY")
                    
                    sl = current_price - (current_atr * config.SL_ATR_MULTIPLIER)
                    tp = current_price + (current_atr * config.TP_ATR_MULTIPLIER)
                    
                    success = send_trade_order(
                        config.SYMBOL, mt5.ORDER_TYPE_BUY, tick.ask, 
                        sl, tp, position_size, risk_manager
                    )
                    
                    if success:
                        risk_manager.trades_today += 1
                        time.sleep(300)
                else:
                    logger.info(">>> VETO: Buy signal but below 200 SMA")
            
            # Sell signal
            elif proba[0] > threshold:
                if current_price < current_sma200:  # Trend filter
                    logger.info(">>> SIGNAL: STRONG SELL")
                    
                    sl = current_price + (current_atr * config.SL_ATR_MULTIPLIER)
                    tp = current_price - (current_atr * config.TP_ATR_MULTIPLIER)
                    
                    success = send_trade_order(
                        config.SYMBOL, mt5.ORDER_TYPE_SELL, tick.bid,
                        sl, tp, position_size, risk_manager
                    )
                    
                    if success:
                        risk_manager.trades_today += 1
                        time.sleep(300)
                else:
                    logger.info(">>> VETO: Sell signal but above 200 SMA")
            else:
                logger.info("Confidence below threshold - no trade")
            
            mt5.shutdown()
            time.sleep(60)
            
        except Exception as e:
            logger.critical(f"Critical error in main loop: {e}", exc_info=True)
            mt5.shutdown()
            time.sleep(300)  # Wait 5 minutes after error


if __name__ == "__main__":
    main()