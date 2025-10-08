"""
Feature Engineering Module
Generates technical indicators and features for ML models
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


class FeatureEngineer:
    def __init__(self, lookback_periods=[5, 10, 20, 50]):
        """
        Initialize feature engineer
        
        Args:
            lookback_periods: List of periods for rolling calculations
        """
        self.lookback_periods = lookback_periods
        
    def add_price_features(self, df):
        """Add price-based features"""
        df = df.copy()
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['oc_range'] = (df['open'] - df['close']).abs() / df['close']
        
        # Price position within candle
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Returns over different periods
        for period in self.lookback_periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'high_{period}'] = df['high'].rolling(period).max()
            df[f'low_{period}'] = df['low'].rolling(period).min()
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        
        return df
    
    def add_trend_indicators(self, df):
        """Add trend-following indicators"""
        df = df.copy()
        
        # Simple Moving Averages
        for period in [7, 14, 21, 50]:
            sma = SMAIndicator(close=df['close'], window=period)
            df[f'sma_{period}'] = sma.sma_indicator()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
        
        # Exponential Moving Averages
        for period in [12, 26]:
            ema = EMAIndicator(close=df['close'], window=period)
            df[f'ema_{period}'] = ema.ema_indicator()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # ADX (trend strength)
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
        df['adx'] = adx.adx()
        
        return df
    
    def add_momentum_indicators(self, df):
        """Add momentum indicators"""
        df = df.copy()
        
        # RSI
        for period in [14, 21]:
            rsi = RSIIndicator(close=df['close'], window=period)
            df[f'rsi_{period}'] = rsi.rsi()
        
        # Stochastic
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Rate of Change
        for period in [9, 14]:
            roc = ROCIndicator(close=df['close'], window=period)
            df[f'roc_{period}'] = roc.roc()
        
        return df
    
    def add_volatility_indicators(self, df):
        """Add volatility indicators"""
        df = df.copy()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-10)
        
        # Average True Range
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
        df['atr'] = atr.average_true_range()
        df['atr_percent'] = df['atr'] / df['close']
        
        return df
    
    def add_volume_indicators(self, df):
        """Add volume-based indicators"""
        df = df.copy()
        
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        
        # Volume moving averages
        for period in [7, 14, 21]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
        
        # On Balance Volume
        obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['obv'] = obv.on_balance_volume()
        df['obv_change'] = df['obv'].pct_change()
        
        # VWAP (if we have enough data)
        if len(df) >= 14:
            vwap = VolumeWeightedAveragePrice(
                high=df['high'], 
                low=df['low'], 
                close=df['close'], 
                volume=df['volume']
            )
            df['vwap'] = vwap.volume_weighted_average_price()
            df['price_to_vwap'] = df['close'] / df['vwap']
        
        return df
    
    def add_time_features(self, df):
        """Add time-based features"""
        df = df.copy()
        
        # Extract time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        
        # Cyclical encoding for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def add_candle_patterns(self, df):
        """
        Add candlestick pattern recognition features
        
        Patterns detected:
        - Doji: Open ≈ Close (indecision)
        - Hammer: Long lower wick, small body at top (bullish reversal)
        - Shooting Star: Long upper wick, small body at bottom (bearish reversal)
        - Bullish Engulfing: Big green candle engulfs previous red candle
        - Bearish Engulfing: Big red candle engulfs previous green candle
        """
        df = df.copy()
        
        # Calculate candle body and wicks
        body = abs(df['close'] - df['open'])
        body_pct = body / df['close']
        upper_wick = df['high'] - np.maximum(df['open'], df['close'])
        lower_wick = np.minimum(df['open'], df['close']) - df['low']
        total_range = df['high'] - df['low']
        
        # Avoid division by zero
        total_range = total_range.replace(0, np.nan)
        
        # 1. DOJI - Open ≈ Close (indecision)
        # Body is very small relative to range
        df['pattern_doji'] = (body_pct < 0.001).astype(int)
        
        # 2. HAMMER - Long lower wick, small body at top (bullish reversal)
        # Lower wick > 2x body, upper wick small, body in upper half
        df['pattern_hammer'] = (
            (lower_wick > 2 * body) &
            (upper_wick < body) &
            (body < 0.3 * total_range)
        ).astype(int)
        
        # 3. SHOOTING STAR - Long upper wick, small body at bottom (bearish reversal)
        # Upper wick > 2x body, lower wick small, body in lower half
        df['pattern_shooting_star'] = (
            (upper_wick > 2 * body) &
            (lower_wick < body) &
            (body < 0.3 * total_range)
        ).astype(int)
        
        # 4. BULLISH ENGULFING - Green candle engulfs previous red candle
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        prev_body = abs(prev_close - prev_open)
        
        df['pattern_bullish_engulfing'] = (
            (df['close'] > df['open']) &  # Current is green
            (prev_close < prev_open) &    # Previous was red
            (df['open'] < prev_close) &   # Current opens below prev close
            (df['close'] > prev_open) &   # Current closes above prev open
            (body > prev_body)             # Current body bigger than previous
        ).astype(int)
        
        # 5. BEARISH ENGULFING - Red candle engulfs previous green candle
        df['pattern_bearish_engulfing'] = (
            (df['close'] < df['open']) &  # Current is red
            (prev_close > prev_open) &    # Previous was green
            (df['open'] > prev_close) &   # Current opens above prev close
            (df['close'] < prev_open) &   # Current closes below prev open
            (body > prev_body)             # Current body bigger than previous
        ).astype(int)
        
        # Additional pattern strength indicators
        df['body_to_range'] = body / total_range
        df['upper_wick_ratio'] = upper_wick / total_range
        df['lower_wick_ratio'] = lower_wick / total_range
        
        return df
    
    def add_support_resistance(self, df, lookback=50):
        """
        Add support and resistance level features
        
        Features:
        - Distance to recent support (lowest low in lookback)
        - Distance to recent resistance (highest high in lookback)
        - Number of times price touched support/resistance
        - Trend line slope
        """
        df = df.copy()
        
        # Rolling support and resistance levels
        df['support_level'] = df['low'].rolling(window=lookback, min_periods=5).min()
        df['resistance_level'] = df['high'].rolling(window=lookback, min_periods=5).max()
        
        # Distance to support/resistance (as % of price)
        df['distance_to_support'] = (df['close'] - df['support_level']) / df['close']
        df['distance_to_resistance'] = (df['resistance_level'] - df['close']) / df['close']
        
        # Position between support and resistance (0 = at support, 1 = at resistance)
        range_sr = df['resistance_level'] - df['support_level']
        df['position_in_range'] = (df['close'] - df['support_level']) / (range_sr + 1e-10)
        
        # Trend line (linear regression slope over lookback period)
        # Positive slope = uptrend, negative = downtrend
        def calculate_trend_slope(prices):
            """Calculate linear regression slope"""
            if len(prices) < 2:
                return 0
            x = np.arange(len(prices))
            y = prices.values
            if len(y) == 0 or np.isnan(y).all():
                return 0
            # Remove NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return 0
            x_clean = x[mask]
            y_clean = y[mask]
            # Linear regression
            slope = np.polyfit(x_clean, y_clean, 1)[0]
            return slope
        
        df['trend_slope'] = df['close'].rolling(window=lookback, min_periods=10).apply(
            calculate_trend_slope, raw=False
        )
        
        # Normalize slope by price (% change per candle)
        df['trend_slope_pct'] = df['trend_slope'] / df['close']
        
        # Support/Resistance strength (how many times price bounced off levels)
        # Count touches within 0.5% of support
        tolerance = 0.005
        df['support_touches'] = 0
        df['resistance_touches'] = 0
        
        for i in range(lookback, len(df)):
            support = df['support_level'].iloc[i]
            resistance = df['resistance_level'].iloc[i]
            
            # Count touches in lookback window
            window_lows = df['low'].iloc[i-lookback:i]
            window_highs = df['high'].iloc[i-lookback:i]
            
            support_touches = ((window_lows - support).abs() / support < tolerance).sum()
            resistance_touches = ((window_highs - resistance).abs() / resistance < tolerance).sum()
            
            df.loc[df.index[i], 'support_touches'] = support_touches
            df.loc[df.index[i], 'resistance_touches'] = resistance_touches
        
        # Near support/resistance flags
        df['near_support'] = (df['distance_to_support'] < 0.01).astype(int)  # Within 1%
        df['near_resistance'] = (df['distance_to_resistance'] < 0.01).astype(int)
        
        return df
    
    def add_market_context_features(self, df, order_book_context=None, ticker_context=None, 
                                    funding_rate=None, long_short_ratio=None):
        """Add market context features from order book, ticker, and sentiment data"""
        df = df.copy()
        
        if order_book_context:
            # Order book imbalance (bid/ask pressure)
            df['bid_ask_imbalance'] = order_book_context.get('order_book_imbalance', 0)
            df['buy_pressure'] = order_book_context.get('buy_pressure', 0.5)
            df['spread_percent'] = order_book_context['spread'] / df['close'].iloc[-1]
            
            # Legacy support
            if 'order_book_imbalance' in order_book_context:
                df['order_book_imbalance'] = order_book_context['order_book_imbalance']
        else:
            # Default values when context not available (historical training)
            df['bid_ask_imbalance'] = 0.0
            df['buy_pressure'] = 0.5
            df['spread_percent'] = 0.0
        
        if ticker_context:
            df['price_change_24h'] = ticker_context['price_change_percent']
            df['high_low_range_24h'] = (
                ticker_context['high_price'] - ticker_context['low_price']
            ) / ticker_context['low_price']
        else:
            # Default values when context not available
            df['price_change_24h'] = 0.0
            df['high_low_range_24h'] = 0.0
        
        # Market sentiment features (futures market)
        if funding_rate is not None:
            df['funding_rate'] = funding_rate
            # Categorize funding rate strength
            df['funding_bullish'] = (funding_rate > 0.0001).astype(int)  # Positive funding
            df['funding_bearish'] = (funding_rate < -0.0001).astype(int)  # Negative funding
        else:
            # Default values when sentiment not available
            df['funding_rate'] = 0.0
            df['funding_bullish'] = 0
            df['funding_bearish'] = 0
        
        if long_short_ratio is not None:
            df['long_short_ratio'] = long_short_ratio
            # Categorize sentiment
            df['sentiment_bullish'] = (long_short_ratio > 1.2).astype(int)  # Strong long bias
            df['sentiment_bearish'] = (long_short_ratio < 0.8).astype(int)  # Strong short bias
        else:
            # Default values when sentiment not available
            df['long_short_ratio'] = 1.0
            df['sentiment_bullish'] = 0
            df['sentiment_bearish'] = 0
        
        return df
    
    def create_target_labels(self, df, threshold_percent=0.5, lookahead_periods=6):
        """
        Create target labels for price direction (binary classification)
        
        Args:
            df: DataFrame with OHLCV data
            threshold_percent: Threshold for "significant" movement metadata (not used for classification)
            lookahead_periods: How many periods ahead to look for the movement
            
        Returns:
            DataFrame with target labels:
            - 0: Downward direction (price will go down more than up)
            - 1: Upward direction (price will go up more than down)
            
        Note: Always predicts a direction. Use confidence + magnitude for action decisions.
        """
        df = df.copy()
        
        # Calculate future returns
        future_highs = df['high'].rolling(window=lookahead_periods, min_periods=1).max().shift(-lookahead_periods)
        future_lows = df['low'].rolling(window=lookahead_periods, min_periods=1).min().shift(-lookahead_periods)
        
        # Calculate percentage moves from current close
        upward_move = ((future_highs - df['close']) / df['close']) * 100
        downward_move = ((df['close'] - future_lows) / df['close']) * 100
        
        # Binary classification: which direction has more movement?
        # 1 = Upward direction (up movement > down movement)
        # 0 = Downward direction (down movement >= up movement)
        df['target'] = (upward_move > downward_move).astype(int)
        
        # Store movement magnitudes for analysis
        df['expected_up_move'] = upward_move
        df['expected_down_move'] = downward_move
        df['expected_max_move'] = np.maximum(upward_move, downward_move)
        
        # Store net direction strength
        df['direction_strength'] = upward_move - downward_move  # Positive = bullish, Negative = bearish
        
        return df
    
    def create_regression_target(self, df, lookahead_periods=6):
        """
        Create regression target: predict actual % price change
        
        Args:
            df: DataFrame with OHLCV data
            lookahead_periods: How many periods ahead to look
            
        Returns:
            DataFrame with regression target:
            - target_return: % price change (e.g., +2.5%, -1.8%)
            - Positive = price went up
            - Negative = price went down
        """
        df = df.copy()
        
        # Calculate future close price
        future_close = df['close'].shift(-lookahead_periods)
        
        # Calculate percentage return
        # (future_close - current_close) / current_close * 100
        df['target_return'] = ((future_close - df['close']) / df['close']) * 100
        
        # Also calculate max favorable move (for position sizing)
        future_high = df['high'].rolling(window=lookahead_periods, min_periods=1).max().shift(-lookahead_periods)
        future_low = df['low'].rolling(window=lookahead_periods, min_periods=1).min().shift(-lookahead_periods)
        
        df['max_upside'] = ((future_high - df['close']) / df['close']) * 100
        df['max_downside'] = ((df['close'] - future_low) / df['close']) * 100
        
        return df
    
    def generate_all_features(self, df, order_book_context=None, ticker_context=None,
                            funding_rate=None, long_short_ratio=None,
                            create_targets=True, threshold_percent=0.5, lookahead_periods=6):
        """
        Generate all features at once
        
        Args:
            df: DataFrame with OHLCV data
            order_book_context: Optional order book context dict
            ticker_context: Optional ticker context dict
            funding_rate: Optional funding rate (market sentiment)
            long_short_ratio: Optional long/short ratio (market sentiment)
            create_targets: Whether to create target labels
            threshold_percent: Threshold for large movements
            lookahead_periods: Lookahead for target creation
            
        Returns:
            DataFrame with all features
        """
        print("Generating features...")
        
        # Apply all feature engineering steps
        df = self.add_price_features(df)
        print("  ✓ Price features")
        
        df = self.add_trend_indicators(df)
        print("  ✓ Trend indicators")
        
        df = self.add_momentum_indicators(df)
        print("  ✓ Momentum indicators")
        
        df = self.add_volatility_indicators(df)
        print("  ✓ Volatility indicators")
        
        df = self.add_volume_indicators(df)
        print("  ✓ Volume indicators")
        
        df = self.add_time_features(df)
        print("  ✓ Time features")
        
        # NEW: Candlestick patterns
        df = self.add_candle_patterns(df)
        print("  ✓ Candlestick patterns (Doji, Hammer, Shooting Star, Engulfing)")
        
        # NEW: Support/Resistance levels
        df = self.add_support_resistance(df)
        print("  ✓ Support/Resistance levels & trend lines")
        
        # Add market context features (order book + sentiment)
        df = self.add_market_context_features(
            df, 
            order_book_context=order_book_context, 
            ticker_context=ticker_context,
            funding_rate=funding_rate,
            long_short_ratio=long_short_ratio
        )
        print("  ✓ Market context + sentiment features")
        
        if create_targets:
            df = self.create_target_labels(df, threshold_percent, lookahead_periods)
            print("  ✓ Target labels created")
        
        # Drop rows with NaN values (from rolling calculations)
        initial_rows = len(df)
        df = df.dropna()
        print(f"  ✓ Removed {initial_rows - len(df)} rows with NaN values")
        
        print(f"Final feature set: {len(df)} rows × {len(df.columns)} features")
        
        return df
    
    def get_feature_columns(self, df):
        """Get list of feature columns (excluding OHLCV and target columns)"""
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'target', 'expected_up_move', 'expected_down_move', 'expected_max_move',
                       'direction_strength']  # Exclude - this IS the target (data leakage!)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def calculate_trend_indicators(self, df):
        """
        Calculate short-term and long-term trend indicators
        
        Returns:
            dict with trend analysis:
            - short_term_trend: 'bullish', 'bearish', 'neutral'
            - long_term_trend: 'bullish', 'bearish', 'neutral'
            - trend_strength: 'strong', 'moderate', 'weak'
            - trend_metrics: dict of underlying values
        """
        if df is None or len(df) == 0:
            return {
                'short_term_trend': 'neutral',
                'long_term_trend': 'neutral',
                'trend_strength': 'weak',
                'trend_metrics': {}
            }
        
        latest = df.iloc[-1]
        
        # Short-term trend (SMA 7 vs 14)
        short_sma_7 = latest.get('sma_7', latest['close'])
        short_sma_14 = latest.get('sma_14', latest['close'])
        
        # Long-term trend (SMA 21 vs 50)
        long_sma_21 = latest.get('sma_21', latest['close'])
        long_sma_50 = latest.get('sma_50', latest['close'])
        
        current_price = latest['close']
        
        # Calculate trend percentages
        short_trend_pct = ((short_sma_7 - short_sma_14) / short_sma_14) * 100 if short_sma_14 != 0 else 0
        long_trend_pct = ((long_sma_21 - long_sma_50) / long_sma_50) * 100 if long_sma_50 != 0 else 0
        
        # Price vs SMA positions
        price_vs_sma_14 = ((current_price - short_sma_14) / short_sma_14) * 100 if short_sma_14 != 0 else 0
        price_vs_sma_50 = ((current_price - long_sma_50) / long_sma_50) * 100 if long_sma_50 != 0 else 0
        
        # Determine short-term trend
        if short_trend_pct > 0.1 and price_vs_sma_14 > 0:
            short_term_trend = 'bullish'
        elif short_trend_pct < -0.1 and price_vs_sma_14 < 0:
            short_term_trend = 'bearish'
        else:
            short_term_trend = 'neutral'
        
        # Determine long-term trend
        if long_trend_pct > 0.15 and price_vs_sma_50 > 0:
            long_term_trend = 'bullish'
        elif long_trend_pct < -0.15 and price_vs_sma_50 < 0:
            long_term_trend = 'bearish'
        else:
            long_term_trend = 'neutral'
        
        # Calculate trend strength based on ADX
        adx = latest.get('adx', 0)
        if adx > 40:
            trend_strength = 'strong'
        elif adx > 25:
            trend_strength = 'moderate'
        else:
            trend_strength = 'weak'
        
        # MACD confirmation
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        macd_bullish = macd > macd_signal
        
        return {
            'short_term_trend': short_term_trend,
            'long_term_trend': long_term_trend,
            'trend_strength': trend_strength,
            'trend_metrics': {
                'short_trend_pct': round(short_trend_pct, 4),
                'long_trend_pct': round(long_trend_pct, 4),
                'price_vs_sma_14': round(price_vs_sma_14, 4),
                'price_vs_sma_50': round(price_vs_sma_50, 4),
                'adx': round(adx, 2),
                'macd_bullish': macd_bullish
            }
        }
    
    def generate_contextual_tags(self, df, thresholds=None):
        """
        Generate contextual tags based on market conditions
        
        Args:
            df: DataFrame with feature data
            thresholds: Optional dict of custom thresholds
            
        Returns:
            list of tags: ['high_volatility', 'overbought', 'momentum_gain', etc.]
        """
        if df is None or len(df) == 0:
            return []
        
        # Default thresholds (can be overridden)
        if thresholds is None:
            thresholds = {
                'high_volatility_atr_pct': 1.5,  # ATR % above this = high volatility
                'very_high_volatility_atr_pct': 2.5,
                'overbought_rsi': 70,
                'oversold_rsi': 30,
                'strong_momentum_roc': 2.0,  # ROC % above this = strong momentum
                'high_volume_ratio': 1.5,  # Volume vs MA ratio
                'tight_bollinger': 0.015,  # BB width % below this = squeeze
                'wide_bollinger': 0.035,  # BB width % above this = expansion
            }
        
        latest = df.iloc[-1]
        tags = []
        
        # Volatility tags
        atr_pct = latest.get('atr_percent', 0)
        if atr_pct > thresholds['very_high_volatility_atr_pct']:
            tags.append('very_high_volatility')
        elif atr_pct > thresholds['high_volatility_atr_pct']:
            tags.append('high_volatility')
        else:
            tags.append('low_volatility')
        
        # RSI tags
        rsi_14 = latest.get('rsi_14', 50)
        if rsi_14 > thresholds['overbought_rsi']:
            tags.append('overbought')
        elif rsi_14 < thresholds['oversold_rsi']:
            tags.append('oversold')
        
        # Momentum tags
        roc_14 = latest.get('roc_14', 0)
        if abs(roc_14) > thresholds['strong_momentum_roc']:
            if roc_14 > 0:
                tags.append('momentum_gain')
            else:
                tags.append('momentum_loss')
        
        # Volume tags
        volume_ratio = latest.get('volume_ratio', 1.0)
        if volume_ratio > thresholds['high_volume_ratio']:
            tags.append('high_volume')
        elif volume_ratio < 0.7:
            tags.append('low_volume')
        
        # Bollinger Bands tags
        bb_width = latest.get('bb_width', 0)
        if bb_width < thresholds['tight_bollinger']:
            tags.append('consolidation')
        elif bb_width > thresholds['wide_bollinger']:
            tags.append('expansion')
        
        # Trend tags from ADX
        adx = latest.get('adx', 0)
        if adx > 40:
            tags.append('strong_trend')
        elif adx < 20:
            tags.append('ranging')
        
        # MACD divergence
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        macd_diff = latest.get('macd_diff', 0)
        
        if macd > macd_signal and macd_diff > 0:
            tags.append('bullish_crossover')
        elif macd < macd_signal and macd_diff < 0:
            tags.append('bearish_crossover')
        
        return tags
    
    def calculate_price_targets(self, df, prediction, confidence, interval='1m', lookahead_periods=6):
        """
        Calculate price targets and time estimates for the prediction
        
        Args:
            df: DataFrame with current market data
            prediction: int (0=down, 1=up)
            confidence: float (model confidence)
            interval: str (candle interval: '1m', '5m', etc.)
            lookahead_periods: int (how many periods to look ahead)
            
        Returns:
            dict with:
            - target_price: float (estimated price target)
            - target_low: float (conservative estimate)
            - target_high: float (optimistic estimate)
            - target_time_minutes: int (estimated time to reach target)
            - target_timestamp: str (ISO timestamp of target time)
            - movement_magnitude: float (expected % move)
        """
        current_price = float(df['close'].iloc[-1])
        
        # Calculate historical volatility for realistic targets
        atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else current_price * 0.002
        atr_pct = (atr / current_price) * 100
        
        # Base movement magnitude (percentage)
        # Use ATR as a guide for realistic short-term moves
        base_magnitude = atr_pct * confidence  # Scale by confidence
        
        # Adjust based on confidence and volatility
        conservative_magnitude = base_magnitude * 0.5
        optimistic_magnitude = base_magnitude * 1.5
        
        # Calculate price targets
        if prediction == 1:  # Upward
            target_price = current_price * (1 + base_magnitude / 100)
            target_low = current_price * (1 + conservative_magnitude / 100)
            target_high = current_price * (1 + optimistic_magnitude / 100)
            movement_magnitude = base_magnitude
        else:  # Downward (prediction == 0)
            target_price = current_price * (1 - base_magnitude / 100)
            target_low = current_price * (1 - optimistic_magnitude / 100)  # Lower = more down
            target_high = current_price * (1 - conservative_magnitude / 100)  # Higher = less down
            movement_magnitude = -base_magnitude
        
        # Time estimate based on interval and lookahead
        interval_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, 
            '30m': 30, '1h': 60, '4h': 240, '1d': 1440
        }.get(interval, 1)
        
        target_time_minutes = interval_minutes * lookahead_periods
        
        # Calculate target timestamp
        from datetime import datetime, timedelta
        target_timestamp = (datetime.now() + timedelta(minutes=target_time_minutes)).isoformat()
        
        return {
            'target_price': round(target_price, 2),
            'target_low': round(target_low, 2),
            'target_high': round(target_high, 2),
            'target_time_minutes': target_time_minutes,
            'target_timestamp': target_timestamp,
            'movement_magnitude': round(movement_magnitude, 3),
            'current_price': round(current_price, 2),
            'confidence': round(confidence, 3)
        }
    
    def generate_trading_suggestion(self, prediction, confidence, trend_info, tags, price_targets=None):
        """
        Generate actionable trading suggestion based on all signals
        
        Args:
            prediction: int (0=down, 1=up) - Binary classification
            confidence: float (model confidence)
            trend_info: dict from calculate_trend_indicators()
            tags: list from generate_contextual_tags()
            price_targets: dict from calculate_price_targets() (optional)
            
        Returns:
            dict with:
            - action: 'BUY', 'SELL', 'HOLD', 'WAIT'
            - conviction: 'high', 'medium', 'low'
            - reasoning: list of reasons
            - risk_level: 'low', 'medium', 'high'
            - price_targets: dict with target prices and timeframes
        """
        reasoning = []
        action = 'HOLD'
        conviction = 'low'
        risk_level = 'medium'
        
        # Base action from binary prediction (0=Down, 1=Up)
        if prediction == 1:
            base_action = 'BUY'
            direction = 'upward'
            reasoning.append(f"Model predicts upward direction ({confidence:.1%} confidence)")
        else:  # prediction == 0
            base_action = 'SELL'
            direction = 'downward'
            reasoning.append(f"Model predicts downward direction ({confidence:.1%} confidence)")
        
        # Add price target context if available
        if price_targets:
            target_price = price_targets['target_price']
            target_time = price_targets['target_time_minutes']
            magnitude = abs(price_targets['movement_magnitude'])
            
            if target_time < 60:
                time_str = f"{target_time} minutes"
            elif target_time < 1440:
                time_str = f"{target_time // 60} hours"
            else:
                time_str = f"{target_time // 1440} days"
            
            reasoning.append(f"Target: ${target_price:,.2f} ({magnitude:+.2f}%) in {time_str}")
        risk_level = 'medium'
        
        # Base action from prediction
        if prediction == 1:
            base_action = 'BUY'
            reasoning.append(f"Model predicts upward movement ({confidence:.1%} confidence)")
        elif prediction == 2:
            base_action = 'SELL'
            reasoning.append(f"Model predicts downward movement ({confidence:.1%} confidence)")
        else:
            base_action = 'HOLD'
            reasoning.append(f"No significant movement predicted")
        
        # Adjust based on confidence
        confidence_boost = 0
        if confidence > 0.7:
            conviction = 'high'
            confidence_boost = 2
        elif confidence > 0.5:
            conviction = 'medium'
            confidence_boost = 1
        else:
            conviction = 'low'
            confidence_boost = 0
        
        # Trend alignment check
        trend_score = 0
        short_trend = trend_info.get('short_term_trend', 'neutral')
        long_trend = trend_info.get('long_term_trend', 'neutral')
        trend_strength = trend_info.get('trend_strength', 'weak')
        
        if base_action == 'BUY':
            if short_trend == 'bullish':
                trend_score += 1
                reasoning.append("Short-term trend is bullish")
            if long_trend == 'bullish':
                trend_score += 1
                reasoning.append("Long-term trend is bullish")
            if trend_strength == 'strong':
                trend_score += 1
                reasoning.append("Trend strength is strong")
        elif base_action == 'SELL':
            if short_trend == 'bearish':
                trend_score += 1
                reasoning.append("Short-term trend is bearish")
            if long_trend == 'bearish':
                trend_score += 1
                reasoning.append("Long-term trend is bearish")
            if trend_strength == 'strong':
                trend_score += 1
                reasoning.append("Trend strength is strong")
        
        # Tag-based adjustments
        tag_warnings = []
        tag_confirmations = []
        
        if 'very_high_volatility' in tags:
            risk_level = 'high'
            tag_warnings.append("Very high volatility - increased risk")
        elif 'high_volatility' in tags:
            risk_level = 'medium'
            tag_warnings.append("High volatility present")
        
        if 'overbought' in tags and base_action == 'BUY':
            tag_warnings.append("Market is overbought - reversal risk")
            trend_score -= 1
        elif 'oversold' in tags and base_action == 'SELL':
            tag_warnings.append("Market is oversold - bounce risk")
            trend_score -= 1
        
        if 'momentum_gain' in tags and base_action == 'BUY':
            tag_confirmations.append("Positive momentum confirms buy signal")
            trend_score += 1
        elif 'momentum_loss' in tags and base_action == 'SELL':
            tag_confirmations.append("Negative momentum confirms sell signal")
            trend_score += 1
        
        if 'bullish_crossover' in tags and base_action == 'BUY':
            tag_confirmations.append("Bullish MACD crossover")
            trend_score += 1
        elif 'bearish_crossover' in tags and base_action == 'SELL':
            tag_confirmations.append("Bearish MACD crossover")
            trend_score += 1
        
        if 'consolidation' in tags:
            tag_warnings.append("Market in consolidation - breakout pending")
        
        if 'ranging' in tags:
            tag_warnings.append("No clear trend - ranging market")
            if base_action != 'HOLD':
                trend_score -= 1
        
        # Final decision logic
        total_score = confidence_boost + trend_score
        
        if base_action == 'HOLD' or total_score < 1:
            action = 'WAIT'
            reasoning.append("Signals not strong enough for entry")
            conviction = 'low'
        elif total_score >= 4 and len(tag_warnings) == 0:
            action = base_action
            conviction = 'high'
            reasoning.append("Strong confluence of signals")
        elif total_score >= 2:
            action = base_action
            conviction = 'medium'
        else:
            action = 'WAIT'
            conviction = 'low'
            reasoning.append("Conflicting signals - wait for clarity")
        
        # Add tag reasoning
        reasoning.extend(tag_confirmations)
        reasoning.extend(tag_warnings)
        
        return {
            'action': action,
            'conviction': conviction,
            'reasoning': reasoning,
            'risk_level': risk_level,
            'score_breakdown': {
                'confidence_boost': confidence_boost,
                'trend_score': trend_score,
                'total_score': total_score
            }
        }
