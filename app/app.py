import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import numpy as np
import requests
from dotenv import load_dotenv
import json
import glob

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_scraper import DataScraper

load_dotenv()

# API Configuration - Updated for deployed service
PREDICTION_API_URL = os.getenv('PREDICTION_API_URL', 'https://btc-forecast-api.onrender.com')
BACKEND_API_URL = os.getenv('BACKEND_API_URL', 'https://btc-forecast-api.onrender.com')

# Page configuration
st.set_page_config(
    page_title="Bitcoin AI Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-overlay {
        background-color: rgba(255, 193, 7, 0.1);
        border: 2px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .bullish { color: #28a745; font-weight: bold; }
    .bearish { color: #dc3545; font-weight: bold; }
    .neutral { color: #6c757d; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state first
if 'scraper' not in st.session_state:
    st.session_state.scraper = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()
if 'live_data' not in st.session_state:
    st.session_state.live_data = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False
if 'api_status' not in st.session_state:
    st.session_state.api_status = {'prediction_api': False, 'backend_api': False}
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Auto-initialize on first load
if not st.session_state.initialized:
    try:
        # Default to Bitcoin
        st.session_state.scraper = DataScraper(symbol="BTCUSDT", interval="5m")
        # Auto-fetch historical data
        df = st.session_state.scraper.fetch_historical_("1 day ago UTC")
        st.session_state.historical_data = df
        st.session_state.initialized = True
    except Exception as e:
        pass  # Fail silently, user can see connection status


# Helper Functions
def get_latest_model_stats():
    """Load statistics from the latest trained model"""
    try:
        # Find the latest model directory
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
        model_folders = glob.glob(os.path.join(model_dir, 'BTCUSDT_*'))
        
        if not model_folders:
            return None
        
        # Get the most recent model folder
        latest_model = max(model_folders, key=os.path.getmtime)
        metadata_path = os.path.join(latest_model, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract key stats
        stats = {
            'model_name': os.path.basename(latest_model),
            'training_date': metadata.get('training_date', 'Unknown'),
            'interval': metadata.get('interval', 'Unknown'),
            'n_features': metadata.get('n_features', 0),
            'sequence_length': metadata.get('sequence_length', 0)
        }
        
        # Add performance metrics if available
        if 'performance' in metadata:
            perf = metadata['performance']
            if 'train' in perf:
                stats['train_accuracy'] = perf['train']['accuracy']
                stats['train_f1'] = perf['train']['f1_macro']
                stats['train_roc_auc'] = perf['train']['roc_auc_ovr']
                stats['train_precision'] = perf['train'].get('precision_macro', 0)
                stats['train_recall'] = perf['train'].get('recall_macro', 0)
            
            if 'test' in perf:
                stats['test_accuracy'] = perf['test']['accuracy']
                stats['test_f1'] = perf['test']['f1_macro']
                stats['test_roc_auc'] = perf['test']['roc_auc_ovr']
                stats['test_precision'] = perf['test'].get('precision_macro', 0)
                stats['test_recall'] = perf['test'].get('recall_macro', 0)
            
            if 'cross_validation' in perf:
                cv = perf['cross_validation']
                stats['cv_accuracy_mean'] = cv.get('accuracy_mean', 0)
                stats['cv_accuracy_std'] = cv.get('accuracy_std', 0)
                stats['cv_f1_mean'] = cv.get('f1_mean', 0)
                stats['cv_f1_std'] = cv.get('f1_std', 0)
            
            if 'overfitting_analysis' in perf:
                ov = perf['overfitting_analysis']
                stats['accuracy_gap'] = ov.get('train_test_accuracy_gap', 0)
                stats['f1_gap'] = ov.get('train_test_f1_gap', 0)
                stats['overfitting_severity'] = ov.get('severity', 'Unknown')
        
        return stats
        
    except Exception as e:
        print(f"Error loading model stats: {e}")
        return None


def check_api_health():
    """Check if the deployed prediction API is running"""
    status = {'prediction_api': False, 'backend_api': False}
    
    try:
        response = requests.get(f"{PREDICTION_API_URL}/", timeout=5)
        if response.status_code == 200:
            # Check if models are loaded
            health_data = response.json()
            status['prediction_api'] = health_data.get('model_loaded', False)
            status['backend_api'] = True  # Same service
    except Exception as e:
        print(f"API health check failed: {e}")
        pass
    
    return status


def get_ml_prediction(symbol, interval):
    """Get prediction from deployed ML API"""
    try:
        # Call the deployed API directly
        response = requests.post(
            f"{PREDICTION_API_URL}/predict",
            json={"symbol": symbol, "interval": interval, "use_live_data": True},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Convert to frontend format
            prediction = data['prediction']
            if prediction == 1:
                signal = 'BUY'
            elif prediction == 2:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'signal': signal,
                'confidence': data['confidence'],
                'prediction_label': data['prediction_label'],
                'probabilities': data['probabilities'],
                'current_price': data['current_price'],
                'expected_movement': data.get('expected_movement'),
                'next_periods': data.get('next_periods', []),
                'timestamp': data.get('timestamp')
            }
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Prediction API error: {str(e)}")
        return None

# Title and description
st.title(" Bitcoin AI Trading Dashboard")
st.markdown("""
> **Real-time Bitcoin analysis with AI-powered price movement predictions**  
> *This is a research tool - no actual trading is executed*
""")

# Check API status on load
if 'api_checked' not in st.session_state:
    st.session_state.api_status = check_api_health()
    st.session_state.api_checked = True

# Display system status banner
status_col1, status_col2 = st.columns(2)

with status_col1:
    if st.session_state.scraper:
        st.success("🟢 Data Connection Active")
    else:
        st.warning("⚪ Data Connection Inactive")

with status_col2:
    if st.session_state.api_status.get('prediction_api', False):
        st.success("🟢 Deployed ML API Online")
    else:
        st.error("🔴 Deployed ML API Offline")

st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Status Check
    if st.button("🔍 Check API Status", use_container_width=True):
        st.session_state.api_status = check_api_health()
        st.rerun()
    
    # Display API status
    if st.session_state.api_status['prediction_api']:
        st.success("✅ Deployed API Online & Models Loaded")
        st.caption("🌐 https://btc-forecast-api.onrender.com")
    else:
        st.error("❌ Deployed API Offline or Models Not Loaded")
        st.caption("API may be starting up - try again in a moment")
    
    st.markdown("---")
    
    # Fixed to Bitcoin only
    symbol = "BTCUSDT"
    st.info(" **Trading Pair:** Bitcoin (BTC/USDT)")
    
    # Time interval selection
    interval = st.selectbox(
        "Time Interval",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
        index=1,
        help="Select the candlestick timeframe"
    )
    
    st.markdown("---")
    
    # Connection status
    if st.session_state.scraper:
        st.success("✅ Connected to Binance")
        if not st.session_state.historical_data.empty:
            st.info(f"📈 {len(st.session_state.historical_data)} data points loaded")
    else:
        st.error("❌ Not connected")
    
    # Refresh data button
    if st.button(" Refresh Data", use_container_width=True):
        if st.session_state.scraper:
            with st.spinner("Refreshing data..."):
                try:
                    st.session_state.scraper = DataScraper(symbol=symbol, interval=interval)
                    df = st.session_state.scraper.fetch_historical_("1 day ago UTC")
                    st.session_state.historical_data = df
                    st.success(f"Refreshed! Loaded {len(df)} data points")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to refresh: {str(e)}")
        else:
            # Try to reconnect
            with st.spinner("Reconnecting..."):
                try:
                    st.session_state.scraper = DataScraper(symbol=symbol, interval=interval)
                    df = st.session_state.scraper.fetch_historical_("1 day ago UTC")
                    st.session_state.historical_data = df
                    st.success("Connected and data loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")

# Main Dashboard
col1, col2 = st.columns([3, 1])

with col1:
    st.header(f"📈 Bitcoin (BTC/USDT) - {interval} Chart")
    
    # Display chart if we have data
    if not st.session_state.historical_data.empty:
        df = st.session_state.historical_data.copy()
        
        # Create candlestick chart
        fig = go.Figure()
        
        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="BTC/USDT",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # Add volume as subplot
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name="Volume",
            yaxis='y2',
            opacity=0.3,
            marker_color='lightblue'
        ))
        
        # Add AI prediction overlay (mock for now)
        if st.session_state.predictions:
            prediction_times = df.index[-len(st.session_state.predictions):]
            fig.add_trace(go.Scatter(
                x=prediction_times,
                y=st.session_state.predictions,
                mode='lines+markers',
                name='AI Prediction',
                line=dict(color='orange', width=3, dash='dot'),
                marker=dict(size=8, symbol='diamond')
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Bitcoin (BTC/USDT) - {interval} Intervals",
            yaxis_title="Price (USDT)",
            yaxis2=dict(title="Volume", overlaying='y', side='right'),
            xaxis_title="Time",
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ML Prediction Button
        col_pred1, col_pred2 = st.columns(2)
        
        with col_pred1:
            if st.button("🤖 Get ML Prediction", use_container_width=True):
                if st.session_state.api_status.get('prediction_api', False):
                    with st.spinner("Generating AI prediction..."):
                        prediction_data = get_ml_prediction("BTCUSDT", interval)
                        
                        if prediction_data:
                            st.session_state.prediction_data = prediction_data
                            
                            # Generate prediction overlay for chart
                            if 'next_periods' in prediction_data and prediction_data['next_periods']:
                                current_price = prediction_data['current_price']
                                next_prices = [p['estimated_price'] for p in prediction_data['next_periods']]
                                st.session_state.predictions = [current_price] + next_prices
                                
                                st.success("✅ Prediction generated!")
                                st.rerun()
                            else:
                                st.warning("Prediction received but no future estimates available")
                        else:
                            st.error("Failed to get prediction from API")
                else:
                    st.error("⚠️ Prediction API is not available.")
                    st.info("The deployed API may be starting up. Please wait a moment and try again.")
                    if st.button("🔄 Retry API Connection"):
                        st.session_state.api_status = check_api_health()
                        st.rerun()
        
        with col_pred2:
            if st.button("🔄 Generate Demo Prediction", use_container_width=True):
                # Fallback: Simulate AI prediction when API not available
                if len(df) > 0:
                    last_prices = df['close'].tail(10).values
                    trend = np.mean(np.diff(last_prices))
                    noise = np.random.normal(0, last_prices[-1] * 0.001, 7)
                    predictions = last_prices[-1] + np.cumsum([trend] * 7) + noise
                    st.session_state.predictions = predictions.tolist()
                    st.info("Demo prediction generated (not from ML model)")
                    st.rerun()
    
    else:
        st.info("� Loading Bitcoin data...")
        st.markdown("If data doesn't appear, click **🔄 Refresh Data** in the sidebar.")

with col2:
    st.header("📊 Market Analytics")
    
    # Model Performance Stats
    st.subheader("🤖 ML Model Stats")
    model_stats = get_latest_model_stats()
    
    if model_stats:
        # Performance Comparison Table
        stats_data = {
            "Metric": [],
            "Training": [],
            "Testing": []
        }
        
        # Add accuracy
        if 'train_accuracy' in model_stats and 'test_accuracy' in model_stats:
            stats_data["Metric"].append("Accuracy")
            stats_data["Training"].append(f"{model_stats['train_accuracy']*100:.2f}%")
            stats_data["Testing"].append(f"{model_stats['test_accuracy']*100:.2f}%")
        
        # Add precision
        if 'train_precision' in model_stats and 'test_precision' in model_stats:
            stats_data["Metric"].append("Precision")
            stats_data["Training"].append(f"{model_stats['train_precision']:.4f}")
            stats_data["Testing"].append(f"{model_stats['test_precision']:.4f}")
        
        # Add recall
        if 'train_recall' in model_stats and 'test_recall' in model_stats:
            stats_data["Metric"].append("Recall")
            stats_data["Training"].append(f"{model_stats['train_recall']:.4f}")
            stats_data["Testing"].append(f"{model_stats['test_recall']:.4f}")
        
        # Add F1 Score
        if 'train_f1' in model_stats and 'test_f1' in model_stats:
            stats_data["Metric"].append("F1 Score")
            stats_data["Training"].append(f"{model_stats['train_f1']:.4f}")
            stats_data["Testing"].append(f"{model_stats['test_f1']:.4f}")
        
        # Add ROC AUC
        if 'train_roc_auc' in model_stats and 'test_roc_auc' in model_stats:
            stats_data["Metric"].append("ROC AUC")
            stats_data["Training"].append(f"{model_stats['train_roc_auc']:.4f}")
            stats_data["Testing"].append(f"{model_stats['test_roc_auc']:.4f}")
        
        # Display the main performance table
        if stats_data["Metric"]:
            st.dataframe(stats_df := pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
            
            # Cross-Validation Results (if available)
            if 'cv_accuracy_mean' in model_stats:
                st.markdown("**📊 K-Fold Cross-Validation**")
                cv_cols = st.columns(2)
                with cv_cols[0]:
                    st.metric(
                        "CV Accuracy", 
                        f"{model_stats['cv_accuracy_mean']*100:.2f}%",
                        delta=f"±{model_stats['cv_accuracy_std']*100:.2f}%"
                    )
                with cv_cols[1]:
                    st.metric(
                        "CV F1 Score",
                        f"{model_stats['cv_f1_mean']:.4f}",
                        delta=f"±{model_stats['cv_f1_std']:.4f}"
                    )
                
                # Variance assessment
                if model_stats['cv_accuracy_std'] < 0.05:
                    st.success("✅ Low variance - Good generalization")
                elif model_stats['cv_accuracy_std'] < 0.10:
                    st.warning("⚠️ Moderate variance")
                else:
                    st.error("🔴 High variance - Unstable model")
            
            # Overfitting Analysis
            st.markdown("**🔍 Overfitting Analysis**")
            if 'accuracy_gap' in model_stats:
                gap_pct = model_stats['accuracy_gap'] * 100
                severity = model_stats.get('overfitting_severity', 'Unknown')
                
                if severity == "severe":
                    st.error(f"🔴 SEVERE Overfitting: {gap_pct:+.2f}% gap")
                    st.caption("⚠️ Model memorizing training data")
                elif severity == "moderate":
                    st.warning(f"⚠️ MODERATE Overfitting: {gap_pct:+.2f}% gap")
                    st.caption("Consider more regularization")
                else:
                    st.success(f"✅ GOOD Fit: {gap_pct:+.2f}% gap")
                    st.caption("Model generalizes well")
                
                # Show regularization status
                st.markdown("**🛡️ Applied Regularization:**")
                st.caption("• L1/L2 Regularization (CatBoost: L2=10, LSTM: L2=0.01)")
                st.caption("• Dropout (LSTM: 0.5, Recurrent: 0.3)")
                st.caption("• Reduced Complexity (Fewer layers/units)")
                st.caption("• K-Fold Cross-Validation (k=5)")
            
            # Model metadata in expander
            with st.expander("📋 Model Details"):
                st.caption(f"**Model:** {model_stats.get('model_name', 'Unknown')}")
                st.caption(f"**Trained:** {model_stats.get('training_date', 'Unknown')}")
                st.caption(f"**Interval:** {model_stats.get('interval', 'Unknown')}")
                st.caption(f"**Features:** {model_stats.get('n_features', 0)}")
                st.caption(f"**Sequence Length:** {model_stats.get('sequence_length', 0)}")
        else:
            st.info("Model stats available but no performance metrics found")
    else:
        st.info("No trained model found. Train a model first using `train_pipeline.py`")
    
    st.markdown("---")
    
    # Real-time market data
    if st.session_state.scraper:
        if st.button("🔄 Update Market Data"):
            with st.spinner("Fetching market data..."):
                try:
                    # Fetch current market context
                    _, context = st.session_state.scraper.fetch_context_data()
                    
                    # Display key metrics
                    ticker_data = context['ticker']
                    order_book_data = context['order_book']
                    
                    # Price change
                    price_change = ticker_data['price_change_percent']
                    color = "🟢" if price_change > 0 else "🔴" if price_change < 0 else "⚪"
                    
                    st.metric(
                        "24h Change", 
                        f"{price_change:.2f}%",
                        delta=f"{price_change:.2f}%"
                    )
                    
                    # Price range
                    st.metric("24h High", f"${ticker_data['high_price']:,.2f}")
                    st.metric("24h Low", f"${ticker_data['low_price']:,.2f}")
                    st.metric("24h Volume", f"{ticker_data['volume']:,.0f}")
                    
                    # Order book metrics
                    st.subheader("📖 Order Book")
                    st.metric("Bid Volume", f"{order_book_data['bid_volume']:,.2f}")
                    st.metric("Ask Volume", f"{order_book_data['ask_volume']:,.2f}")
                    st.metric("Spread", f"${order_book_data['spread']:.2f}")
                    
                    # Market sentiment indicator
                    bid_ask_ratio = order_book_data['bid_volume'] / order_book_data['ask_volume']
                    if bid_ask_ratio > 1.1:
                        sentiment = "🟢 Bullish"
                        sentiment_class = "bullish"
                    elif bid_ask_ratio < 0.9:
                        sentiment = "🔴 Bearish" 
                        sentiment_class = "bearish"
                    else:
                        sentiment = "⚪ Neutral"
                        sentiment_class = "neutral"
                    
                    st.markdown(f"**Market Sentiment:** <span class='{sentiment_class}'>{sentiment}</span>", 
                              unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Failed to fetch market data: {str(e)}")
    
    # AI Prediction Panel
    st.subheader("🤖 AI Predictions")
    
    if st.session_state.prediction_data:
        pred_data = st.session_state.prediction_data
        
        # Main prediction display
        signal_colors = {
            'BUY': 'bullish',
            'SELL': 'bearish',
            'HOLD': 'neutral'
        }
        
        signal_icons = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '⚪'
        }
        
        signal = pred_data.get('signal', 'HOLD')
        confidence = pred_data.get('confidence', 0)
        current_price = pred_data.get('current_price', 0)
        expected_movement = pred_data.get('expected_movement', 0)
        
        # Calculate expected price
        if expected_movement:
            expected_price = current_price * (1 + expected_movement / 100)
        else:
            expected_price = current_price
        
        st.markdown(f"""
        <div class="prediction-overlay">
            <h3>{signal_icons.get(signal, '⚪')} {signal} Signal</h3>
            <p><strong>Prediction:</strong> {pred_data.get('prediction_label', 'Unknown')}</p>
            <p><strong>Confidence:</strong> <span class="{signal_colors.get(signal, 'neutral')}">{confidence:.1%}</span></p>
            <hr>
            <p><strong>Current Price:</strong> ${current_price:,.2f}</p>
            <p><strong>Expected Price:</strong> ${expected_price:,.2f}</p>
            <p><strong>Expected Move:</strong> <span class="{signal_colors.get(signal, 'neutral')}">{expected_movement:+.2f}%</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence meter
        st.progress(confidence)
        st.caption(f"Model Confidence: {confidence:.1%}")
        
        # Probabilities breakdown
        st.markdown("**Class Probabilities:**")
        probs = pred_data.get('probabilities', {})
        
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("No Move", f"{probs.get('no_movement', 0):.1%}")
        with col_p2:
            st.metric("Large Up", f"{probs.get('large_up', 0):.1%}", 
                     delta="Bullish" if probs.get('large_up', 0) > 0.5 else None)
        with col_p3:
            st.metric("Large Down", f"{probs.get('large_down', 0):.1%}",
                     delta="Bearish" if probs.get('large_down', 0) > 0.5 else None)
        
        # Next periods forecast
        if 'next_periods' in pred_data and pred_data['next_periods']:
            st.markdown("**Next Periods Forecast:**")
            periods_df = pd.DataFrame(pred_data['next_periods'])
            
            # Create mini chart for forecast
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=periods_df['period'],
                y=periods_df['estimated_price'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='orange', width=2),
                marker=dict(size=6)
            ))
            
            fig_forecast.update_layout(
                title="Price Forecast",
                xaxis_title="Periods Ahead",
                yaxis_title="Price (USDT)",
                height=250,
                showlegend=False,
                template='plotly_white',
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Timestamp
        st.caption(f"Updated: {pred_data.get('timestamp', 'N/A')}")
        
        # Clear prediction button
        if st.button("🗑️ Clear Prediction", use_container_width=True):
            st.session_state.prediction_data = None
            st.session_state.predictions = []
            st.rerun()
            
    elif st.session_state.predictions:
        # Fallback for demo predictions
        current_price = st.session_state.historical_data['close'].iloc[-1] if not st.session_state.historical_data.empty else 0
        next_prediction = st.session_state.predictions[0] if st.session_state.predictions else current_price
        
        prediction_change = ((next_prediction - current_price) / current_price) * 100
        
        st.markdown(f"""
        <div class="prediction-overlay">
            <h4>Demo Prediction</h4>
            <p><strong>Current:</strong> ${current_price:.2f}</p>
            <p><strong>Predicted:</strong> ${next_prediction:.2f}</p>
            <p><strong>Change:</strong> <span class="{'bullish' if prediction_change > 0 else 'bearish'}">{prediction_change:+.2f}%</span></p>
            <p><em>⚠️ This is a demo - not from ML model</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence meter (mock)
        confidence = np.random.uniform(0.6, 0.95)
        st.progress(confidence)
        st.caption(f"Demo Confidence: {confidence:.1%}")
        
    else:
        st.info("Click '🤖 Get ML Prediction' to see AI analysis")
        
        # Show API status if not available
        if not st.session_state.api_status.get('prediction_api', False):
            st.markdown("---")
            st.markdown("**🌐 Using Deployed API Service**")
            st.caption("API URL: https://btc-forecast-api.onrender.com")
            st.caption("Note: Deployed services may take 30-60 seconds to wake up from sleep")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚠️ <strong>Disclaimer:</strong> This is a research tool for educational purposes only. 
    Not financial advice. Past performance does not guarantee future results.</p>
</div>
""", unsafe_allow_html=True)
