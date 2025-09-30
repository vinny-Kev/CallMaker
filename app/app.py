import streamlit as st
import requests
import pandas as pd
import plotly.graph_objs as go
from binance import ThreadedWebsocketManager
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Crypto Trading Assistant", layout="wide")
st.title("Crypto Trading Assistant (Research Only)")
st.markdown("> **This tool is for research only. No trading is executed.**")

symbol = st.sidebar.selectbox("Symbol", ["BTCUSDT"])  # Add more later
interval = '5m'

# Live data stub (WebSocket)
if 'candles' not in st.session_state:
    st.session_state['candles'] = []

# TODO: Implement real WebSocket streaming
if st.button("Simulate Live Data Update"):
    # Simulate new candle
    if st.session_state['candles']:
        last = st.session_state['candles'][-1]
        new = last.copy()
        new['close'] += 10
        st.session_state['candles'].append(new)
    else:
        st.session_state['candles'] = [{"open": 40000, "high": 40100, "low": 39900, "close": 40050, "volume": 1, "open_time": "2023-01-01T00:00:00"}]

candles = st.session_state['candles']
df = pd.DataFrame(candles)
if not df.empty:
    fig = go.Figure(data=[go.Candlestick(x=df['open_time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
    st.plotly_chart(fig, use_container_width=True)
    # Backend prediction overlay stub
    if st.button("Get Prediction"):
        resp = requests.post("http://localhost:8000/predict", json={"candles": candles})
        if resp.ok:
            result = resp.json()
            st.write(f"**Signal:** {result['signal']} | **Confidence:** {result['confidence']:.2f}")
        else:
            st.write("Prediction service unavailable.")
