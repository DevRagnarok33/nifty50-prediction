import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import requests
import plotly.express as px
import plotly.graph_objects as go

# ── PAGE CONFIG ────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence",
    page_icon="📈",
    layout="wide"
)

# ── LOAD CSV DATA ──────────────────────────────────────
@st.cache_data
def load_data():
    local_path = 'NIFTY50_all.csv'
    if not os.path.exists(local_path):
        try:
            file_id = "1YXM-hBa_2orAI2eMyyE5XE1Qrs4kJtVU"
            session  = requests.Session()
            url      = "https://drive.google.com/uc?export=download"
            response = session.get(url, params={'id': file_id}, stream=True)
            token    = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            if token:
                response = session.get(url, params={'id': file_id, 'confirm': token}, stream=True)
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(32768):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            st.error(f"❌ Could not download dataset: {e}")
            return None
    if not os.path.exists(local_path):
        st.error("❌ Dataset file not found.")
        return None
    df = pd.read_csv(local_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    return df

# ── MARKET RETURN FROM CSV (no yfinance) ──────────────
@st.cache_data
def load_market(_df):
    daily = (
        _df.groupby(['Date'])['Close']
        .mean()
        .pct_change() * 100
    )
    market = daily.reset_index()
    market.columns = ['Date', 'Market_Return']
    return market.dropna().reset_index(drop=True)

# ── COMPUTE STATS ──────────────────────────────────────
@st.cache_data
def compute_stats(_df, _market):
    market = _market.set_index('Date')
    results = []
    for symbol in _df['Symbol'].unique():
        try:
            s = _df[_df['Symbol'] == symbol].copy()
            s['Daily_Return'] = s['Close'].pct_change()
            s = s.dropna(subset=['Daily_Return']).set_index('Date')

            ann_return  = s['Daily_Return'].mean() * 252 * 100
