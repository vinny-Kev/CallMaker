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
    
    def add_market_context_features(self, df, order_book_context=None, ticker_context=None):
        """Add market context features from order book and ticker data"""
        df = df.copy()
        
        if order_book_context:
            df['bid_ask_imbalance'] = order_book_context['bid_volume'] / (
                order_book_context['ask_volume'] + 1e-10
            )
            df['spread_percent'] = order_book_context['spread'] / df['close'].iloc[-1]
        
        if ticker_context:
            df['price_change_24h'] = ticker_context['price_change_percent']
            df['high_low_range_24h'] = (
                ticker_context['high_price'] - ticker_context['low_price']
            ) / ticker_context['low_price']
        
        return df
    
    def create_target_labels(self, df, threshold_percent=0.5, lookahead_periods=6):
        """
        Create target labels for large price movements
        
        Args:
            df: DataFrame with OHLCV data
            threshold_percent: Threshold for "large" movement (0.5 = 0.5%)
            lookahead_periods: How many periods ahead to look for the movement
            
        Returns:
            DataFrame with target labels:
            - 0: No significant movement
            - 1: Large upward movement
            - 2: Large downward movement
        """
        df = df.copy()
        
        # Calculate future returns
        future_highs = df['high'].rolling(window=lookahead_periods, min_periods=1).max().shift(-lookahead_periods)
        future_lows = df['low'].rolling(window=lookahead_periods, min_periods=1).min().shift(-lookahead_periods)
        
        # Calculate percentage moves from current close
        upward_move = ((future_highs - df['close']) / df['close']) * 100
        downward_move = ((df['close'] - future_lows) / df['close']) * 100
        
        # Create labels
        df['target'] = 0  # No significant movement
        df.loc[upward_move >= threshold_percent, 'target'] = 1  # Large up
        df.loc[downward_move >= threshold_percent, 'target'] = 2  # Large down
        
        # Additional features: magnitude of expected movement
        df['expected_up_move'] = upward_move
        df['expected_down_move'] = downward_move
        df['expected_max_move'] = np.maximum(upward_move, downward_move)
        
        return df
    
    def generate_all_features(self, df, order_book_context=None, ticker_context=None,
                            create_targets=True, threshold_percent=0.5, lookahead_periods=6):
        """
        Generate all features at once
        
        Args:
            df: DataFrame with OHLCV data
            order_book_context: Optional order book context dict
            ticker_context: Optional ticker context dict
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
        
        if order_book_context or ticker_context:
            df = self.add_market_context_features(df, order_book_context, ticker_context)
            print("  ✓ Market context features")
        
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
                       'target', 'expected_up_move', 'expected_down_move', 'expected_max_move']
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
    
    def generate_trading_suggestion(self, prediction, confidence, trend_info, tags):
        """
        Generate actionable trading suggestion based on all signals
        
        Args:
            prediction: int (0=no move, 1=up, 2=down)
            confidence: float (model confidence)
            trend_info: dict from calculate_trend_indicators()
            tags: list from generate_contextual_tags()
            
        Returns:
            dict with:
            - action: 'BUY', 'SELL', 'HOLD', 'WAIT'
            - conviction: 'high', 'medium', 'low'
            - reasoning: list of reasons
            - risk_level: 'low', 'medium', 'high'
        """
        reasoning = []
        action = 'HOLD'
        conviction = 'low'
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
