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

# ── TOOLTIPS & DEFINITIONS ─────────────────────────────
TOOLTIPS = {
    "Volatility": "A measure of how much a stock's price fluctuates. Higher volatility means higher risk.",
    "Sharpe": "Measures return for every unit of risk. A ratio > 1.0 is generally considered good.",
    "Sortino": "Similar to Sharpe, but only penalizes 'bad' volatility (downward price drops). Higher is better.",
    "Drawdown": "The largest historical percentage drop in a stock's price from its peak to its lowest point.",
    "Beta": "Measures how sensitive a stock is to the overall market. Beta = 1.0 moves exactly with the market. Beta > 1 is aggressive. Beta < 1 is defensive.",
    "MAE": "Mean Absolute Error: The average error between the predicted price and actual price in Rupees.",
    "RMSE": "Root Mean Squared Error: Similar to MAE, but gives a heavier penalty to very large prediction errors.",
    "R2": "R-Squared: Measures how well the ML model explains the actual price movements. 1.0 is a perfect prediction.",
    "DirAcc": "Directional Accuracy: The percentage of time the model correctly guessed if the stock would go UP or DOWN."
}

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
            ann_vol     = s['Daily_Return'].std()  * np.sqrt(252) * 100
            sharpe      = (ann_return / ann_vol) if ann_vol != 0 else 0
            
            downside_returns = s[s['Daily_Return'] < 0]['Daily_Return']
            down_vol = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 1 else 0
            sortino = (ann_return / down_vol) if down_vol != 0 else 0

            rolling_max = s['Close'].cummax()
            max_dd      = ((s['Close'] - rolling_max) / rolling_max).min() * 100

            combined = s[['Daily_Return']].rename(columns={'Daily_Return': 'Stock_Return'})
            combined['Stock_Return'] *= 100
            combined = combined.join(market[['Market_Return']], how='inner').dropna()

            beta, corr = 0, 0
            if len(combined) > 100:
                cov  = np.cov(combined['Stock_Return'], combined['Market_Return'])
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0
                corr = combined['Stock_Return'].corr(combined['Market_Return'])

            results.append({
                'Symbol'        : symbol,
                'Ann_Return'    : round(ann_return, 2),
                'Ann_Volatility': round(ann_vol, 2),
                'Sharpe_Ratio'  : round(sharpe, 3),
                'Sortino_Ratio' : round(sortino, 3),
                'Max_Drawdown'  : round(max_dd, 2),
                'Beta'          : round(beta, 3),
                'Correlation'   : round(corr, 3)
            })
        except Exception:
            pass
    return pd.DataFrame(results)

# ── LOAD DATA ──────────────────────────────────────────
df = load_data()

if df is None:
    st.error("❌ Could not load data. Please check your Google Drive link.")
    st.stop()

market = load_market(df)
stats  = compute_stats(df, market)

# ── BUILD PORTFOLIOS ───────────────────────────────────
conservative = stats[
    (stats['Ann_Volatility'] < stats['Ann_Volatility'].quantile(0.35)) &
    (stats['Ann_Return'] > 0)
].nlargest(8, 'Sharpe_Ratio')

balanced = stats[
    (stats['Ann_Volatility'] < stats['Ann_Volatility'].quantile(0.65)) &
    (stats['Ann_Return'] > stats['Ann_Return'].quantile(0.35))
].nlargest(8, 'Sharpe_Ratio')

aggressive = stats[
    stats['Ann_Return'] > stats['Ann_Return'].quantile(0.65)
].nlargest(8, 'Ann_Return')

portfolios = {'Conservative': conservative, 'Balanced': balanced, 'Aggressive': aggressive}

# ── SIDEBAR ────────────────────────────────────────────
st.sidebar.image(
    "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=320&auto=format&fit=crop", 
    use_container_width=True
)
st.sidebar.title("NIFTY-50 Platform")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "📊 Stock Analyzer",
    "💼 Portfolio Constructor",
    "⚠️ Risk & Beta Dashboard",
    "🔍 Explainable Recommendations"
])

# ══════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("📈 NIFTY-50 Investment Intelligence Platform")
    st.markdown("### Data-Driven Investment Analysis Using Machine Learning")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",  f"{len(df):,}")
    col2.metric("Stocks Covered", f"{df['Symbol'].nunique()}")
    col3.metric("Date Range",     "2000–2021")
    col4.metric("Years of Data",  "21 Years")

    st.markdown("---")
    
    st.markdown("### 📖 The Beginner's Metric Cheat Sheet")
    st.write("We don't hide the complex math on this platform, but we do make it easy to read. Keep an eye out for these terms as you explore:")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**📉 Volatility:** The roller-coaster metric. High volatility means the price swings wildly. Low means it's stable.")
        st.success("**🛡️ Max Drawdown:** The worst-case scenario. If you bought at the absolute peak, this is how much you would have lost at the absolute bottom.")
    with c2:
        st.warning("**⚖️ Sharpe Ratio:** The efficiency score. It tells you if the returns you are getting are worth the risk you are taking. (Higher is better).")
        st.error("**🛑 Sortino Ratio:** Similar to Sharpe, but it only punishes stocks for dropping in price, ignoring upward swings. (Higher is better).")
    with c3:
        st.info("**🔗 Beta:** The market magnet. A Beta of 1.0 means the stock moves exactly with the market. > 1 means it's aggressive, < 1 means it's defensive.")
    
    st.markdown("---")
    st.markdown(f"### Dataset Overview (Latest Data for All {df['Symbol'].nunique()} Historical Stocks)")
    latest_data = df.groupby('Symbol').last().reset_index()
    st.dataframe(
        latest_data[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']],
        use_container_width=True,
        hide_index=True
    )

# ══════════════════════════════════════════════════════
# PAGE 2 — STOCK ANALYZER
# ══════════════════════════════════════════════════════
elif page == "📊 Stock Analyzer":
    st.title("📊 Stock Analyzer")

    symbol = st.selectbox("Select Stock", sorted(df['Symbol'].unique()))
    stock  = df[df['Symbol'] == symbol].copy()
    stock['Daily_Return'] = stock['Close'].pct_change() * 100
    stock['MA20']         = stock['Close'].rolling(20).mean()
    stock['MA50']         = stock['Close'].rolling(50).mean()
    stock['Volatility']   = stock['Daily_Return'].rolling(30).std()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price",    f"₹{stock['Close'].iloc[-1]:.2f}")
    col2.metric("52W High",         f"₹{stock['Close'].tail(252).max():.2f}")
    col3.metric("52W Low",          f"₹{stock['Close'].tail(252).min():.2f}")
    col4.metric("Avg Daily Return", f"{stock['Daily_Return'].mean():.3f}%", help=TOOLTIPS["Volatility"])

    st.markdown("---")

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=stock['Date'], y=stock['Close'], mode='lines', name='Close Price', line=dict(color='steelblue', width=1.5)))
    fig_price.add_trace(go.Scatter(x=stock['Date'], y=stock['MA20'], mode='lines', name='MA20', line=dict(color='orange', width=2)))
    fig_price.add_trace(go.Scatter(x=stock['Date'], y=stock['MA50'], mode='lines', name='MA50', line=dict(color='red', width=2)))
    fig_price.update_layout(title=f'{symbol} — Price with Moving Averages', xaxis_title='Date', yaxis_title='Price (₹)', hovermode="x unified", height=450)
    st.plotly_chart(fig_price, use_container_width=True)

    with st.expander("📉 View Rolling Volatility Chart"):
        fig_vol = px.line(stock, x='Date', y='Volatility', title=f'{symbol} — 30-Day Rolling Volatility', height=350)
        fig_vol.update_traces(line_color='red', line_width=1.5)
        fig_vol.update_layout(hovermode="x unified", xaxis_title="Date", yaxis_title="Volatility")
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 ML Price Prediction")

    features = ['Lag1', 'Lag2', 'Lag3', 'MA20', 'MA50', 'Volatility', 'Daily_Return']
    stock['Lag1']   = stock['Close'].shift(1)
    stock['Lag2']   = stock['Close'].shift(2)
    stock['Lag3']   = stock['Close'].shift(3)
    stock['Target'] = stock['Close'].shift(-1)
    stock_clean     = stock.dropna()

    if len(stock_clean) < 100:
        st.warning("Not enough data for ML prediction for this stock.")
    else:
        split   = int(len(stock_clean) * 0.8)
        X_train = stock_clean[features].iloc[:split]
        X_test  = stock_clean[features].iloc[split:]
        y_train = stock_clean['Target'].iloc[:split]
        y_test  = stock_clean['Target'].iloc[split:]

        model  = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        
        current_prices_test = stock_clean['Close'].iloc[split:].values
        actual_direction = np.sign(y_test.values - current_prices_test)
        predicted_direction = np.sign(y_pred - current_prices_test)
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE",  f"₹{mae:.2f}", help=TOOLTIPS["MAE"])
        col2.metric("RMSE", f"₹{rmse:.2f}", help=TOOLTIPS["RMSE"])
        col3.metric("R²",   f"{r2:.4f}", help=TOOLTIPS["R2"])
        col4.metric("Dir. Accuracy", f"{directional_accuracy:.1f}%", help=TOOLTIPS["DirAcc"])

        with st.expander("🔬 View Model Accuracy Graph"):
            ml_df = pd.DataFrame({'Trading Days (Test Set)': range(len(y_test)), 'Actual Price': y_test.values, 'Predicted Price': y_pred})
            fig_ml = go.Figure()
            fig_ml.add_trace(go.Scatter(x=ml_df['Trading Days (Test Set)'], y=ml_df['Actual Price'], mode='lines', name='Actual Price', line=dict(color='steelblue', width=2)))
            fig_ml.add_trace(go.Scatter(x=ml_df['Trading Days (Test Set)'], y=ml_df['Predicted Price'], mode='lines', name='Predicted Price', line=dict(color='orange', width=2, dash='dot')))
            fig_ml.update_layout(title=f'{symbol} — Actual vs Predicted Price', xaxis_title='Trading Days (Test Set)', yaxis_title='Price (₹)', hovermode="x unified", height=400)
            st.plotly_chart(fig_ml, use_container_width=True)

        st.markdown("---")
        st.subheader("🔮 Next Day Prediction")
        last_row   = stock_clean[features].iloc[-1]
        next_price = model.predict([last_row])[0]
        last_price = stock_clean['Close'].iloc[-1]
        change     = next_price - last_price
        pct_change = (change / last_price) * 100
        
        direction_label = "Upward 📈" if change > 0 else "Downward 📉"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last Known Price",     f"₹{last_price:.2f}")
        col2.metric("Predicted Next Price", f"₹{next_price:.2f}", f"{change:+.2f}")
        col3.metric("Expected Change",      f"{pct_change:+.2f}%")
        col4.metric("Predicted Direction",  direction_label, help="The forecasted overall trend (Up/Down) for the next trading day.")
        st.caption("⚠️ This prediction is based on historical patterns only. Not financial advice.")

# ══════════════════════════════════════════════════════
# PAGE 3 — PORTFOLIO CONSTRUCTOR
# ══════════════════════════════════════════════════════
elif page == "💼 Portfolio Constructor":
    st.title("💼 Portfolio Constructor")
    st.write("We have automatically built three optimized portfolios based on different risk profiles. Review the top-level stats below.")

    col1, col2, col3 = st.columns(3)
    for col, (name, port), color in zip(
        [col1, col2, col3], portfolios.items(), ['🟢', '🔵', '🔴']
    ):
        with col:
            st.markdown(f"### {color} {name}")
            st.metric("Avg Annual Return", f"{port['Ann_Return'].mean():.1f}%", help="The average expected yearly return.")
            st.metric("Avg Volatility",    f"{port['Ann_Volatility'].mean():.1f}%", help=TOOLTIPS["Volatility"])
            st.metric("Avg Sharpe Ratio",  f"{port['Sharpe_Ratio'].mean():.3f}", help=TOOLTIPS["Sharpe"])
            st.metric("Avg Max Drawdown",  f"{port['Max_Drawdown'].mean():.1f}%", help=TOOLTIPS["Drawdown"])
            st.metric("Avg Beta",          f"{port['Beta'].mean():.3f}", help=TOOLTIPS["Beta"])

    st.markdown("---")
    
    with st.expander("📊 View Visual Allocations & Deep Dive Data", expanded=False):
        st.markdown("### Portfolio Allocations by Investor Profile")
        pie_cols = st.columns(3)
        for col, (name, port) in zip(pie_cols, portfolios.items()):
            weights = [100 / len(port)] * len(port)
            port['Weight'] = weights
            fig_pie = px.pie(port, values='Weight', names='Symbol', hole=0.3,
                             title=f"<b>{name}</b><br><sup>Return: {port['Ann_Return'].mean():.1f}% | Risk: {port['Ann_Volatility'].mean():.1f}%</sup>")
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent')
            fig_pie.update_layout(showlegend=False, margin=dict(t=60, b=20, l=20, r=20))
            col.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        selected = st.selectbox("Select a portfolio to view its exact stock list:", list(portfolios.keys()))
        st.dataframe(
            portfolios[selected][['Symbol', 'Ann_Return', 'Ann_Volatility', 'Sharpe_Ratio', 'Sortino_Ratio', 'Max_Drawdown', 'Beta']]
            .reset_index(drop=True)
            .style.format({
                'Ann_Return'    : '{:.2f}%',
                'Ann_Volatility': '{:.2f}%',
                'Sharpe_Ratio'  : '{:.3f}',
                'Sortino_Ratio' : '{:.3f}',  
                'Max_Drawdown'  : '{:.2f}%',
                'Beta'          : '{:.3f}'
            }),
            use_container_width=True
        )

# ══════════════════════════════════════════════════════
# PAGE 4 — RISK & BETA DASHBOARD
# ══════════════════════════════════════════════════════
elif page == "⚠️ Risk & Beta Dashboard":
    st.title("⚠️ Risk & Beta Dashboard")
    st.write("A quick glance at the extreme ends of the NIFTY-50 market risk spectrum.")

    col1, col2, col3, col4 = st.columns(4)
    safest   = stats.loc[stats['Ann_Volatility'].idxmin()]
    riskiest = stats.loc[stats['Ann_Volatility'].idxmax()]
    best_sh  = stats.loc[stats['Sharpe_Ratio'].idxmax()]
    worst_dd = stats.loc[stats['Max_Drawdown'].idxmin()]

    col1.metric("Safest Stock",   safest['Symbol'],   f"Vol: {safest['Ann_Volatility']:.1f}%", help=TOOLTIPS["Volatility"])
    col2.metric("Riskiest Stock", riskiest['Symbol'], f"Vol: {riskiest['Ann_Volatility']:.1f}%", help=TOOLTIPS["Volatility"])
    col3.metric("Best Sharpe",    best_sh['Symbol'],  f"Sharpe: {best_sh['Sharpe_Ratio']:.3f}", help=TOOLTIPS["Sharpe"])
    col4.metric("Worst Drawdown", worst_dd['Symbol'], f"DD: {worst_dd['Max_Drawdown']:.1f}%", help=TOOLTIPS["Drawdown"])

    st.markdown("---")

    with st.expander("📈 Explore Advanced Risk Charts (Beta, Sharpe, Simulations)", expanded=False):
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        with row1_col1:
            beta_sorted = stats.sort_values('Beta').copy()
            def assign_risk(b):
                if b > 1.2: return 'Aggressive (Red)'
                elif b > 0.8: return 'Moderate (Blue)'
                else: return 'Defensive (Green)'
                
            beta_sorted['Risk_Profile'] = beta_sorted['Beta'].apply(assign_risk)
            
            fig1 = px.bar(beta_sorted, x='Beta', y='Symbol', orientation='h',
                          color='Risk_Profile',
                          color_discrete_map={
                              'Aggressive (Red)': 'tomato',
                              'Moderate (Blue)': 'steelblue',
                              'Defensive (Green)': 'mediumseagreen'
                          },
                          title='Beta — All Stocks', height=500)
            
            fig1.add_vline(x=1.0, line_dash="dash", line_color="black", annotation_text="Market Beta=1.0")
            fig1.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig1, use_container_width=True)

        with row1_col2:
            fig2 = px.scatter(stats, x='Ann_Volatility', y='Ann_Return',
                              color='Sharpe_Ratio', color_continuous_scale='RdYlGn',
                              hover_name='Symbol', size_max=15,
                              title='Risk vs Return', height=500,
                              labels={'Ann_Volatility': 'Volatility (Risk) %', 'Ann_Return': 'Annual Return %'})
            fig2.update_traces(textposition='top center', marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig2, use_container_width=True)

        with row2_col1:
            port_betas = [stats[stats['Symbol'].isin(p['Symbol'])]['Beta'].mean() for p in portfolios.values()]
            port_data = pd.DataFrame({
                'Portfolio': list(portfolios.keys()),
                'Beta': port_betas
            })
            
            fig3 = px.bar(port_data, x='Portfolio', y='Beta', color='Portfolio',
                          color_discrete_sequence=['mediumseagreen', 'steelblue', 'tomato'],
                          text='Beta', title='Average Portfolio Beta', height=400)
            
            fig3.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig3.add_hline(y=1.0, line_dash="dash", line_color="black", annotation_text="Market Beta=1.0")
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with row2_col2:
            market_changes = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
            sim_data = []
            for m in market_changes:
                sim_data.append({'Market Change (%)': m, 'Portfolio Change (%)': m, 'Portfolio': 'Market (β=1.00)'})
            for (name, _), beta in zip(portfolios.items(), port_betas):
                for m in market_changes:
                    sim_data.append({'Market Change (%)': m, 'Portfolio Change (%)': m * beta, 'Portfolio': f'{name} (β={beta:.2f})'})
                    
            sim_df = pd.DataFrame(sim_data)
            fig4 = px.line(sim_df, x='Market Change (%)', y='Portfolio Change (%)', color='Portfolio',
                           markers=True, title='Sensitivity Simulation', height=400)
            
            for trace in fig4.data:
                if 'Market' in trace.name:
                    trace.line.dash = 'dash'
                    trace.line.color = 'gray'
                    trace.mode = 'lines'

            fig4.add_hline(y=0, line_color='black', line_width=1)
            fig4.add_vline(x=0, line_color='black', line_width=1)
            st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════
# PAGE 5 — EXPLAINABLE RECOMMENDATIONS
# ══════════════════════════════════════════════════════
elif page == "🔍 Explainable Recommendations":
    st.title("🔍 Explainable Recommendations")
    st.markdown("Understand exactly **why** each stock was selected for each portfolio. Click on any stock below to see its exact metrics.")

    selected = st.selectbox("Select Portfolio", list(portfolios.keys()))
    port     = portfolios[selected].copy()
    
    port['Beta'] = [stats[stats['Symbol'] == s]['Beta'].values[0] if len(stats[stats['Symbol'] == s]) > 0 else 0 for s in port['Symbol']]

    st.markdown("---")

    for _, row in port.iterrows():
        with st.expander(f"📌 {row['Symbol']}"):
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Annual Return", f"{row['Ann_Return']:.1f}%", help="The expected yearly return based on historical data.")
            col2.metric("Volatility",    f"{row['Ann_Volatility']:.1f}%", help=TOOLTIPS["Volatility"])
            col3.metric("Sharpe Ratio",  f"{row['Sharpe_Ratio']:.3f}", help=TOOLTIPS["Sharpe"])
            col4.metric("Sortino Ratio", f"{row['Sortino_Ratio']:.3f}", help=TOOLTIPS["Sortino"])
            col5.metric("Beta",          f"{row['Beta']:.3f}", help=TOOLTIPS["Beta"])

    st.markdown("---")
    
    with st.expander("🔬 View Visual Market Comparisons (Bar & Radar Charts)", expanded=False):
        bar_col1, bar_col2 = st.columns(2)
        bar_col3, bar_col4 = st.columns(2)
        
        with bar_col1:
            fig_ret = px.bar(port, x='Ann_Return', y='Symbol', orientation='h', title='Annual Return', color_discrete_sequence=['mediumseagreen'], height=350)
            fig_ret.add_vline(x=stats['Ann_Return'].mean(), line_dash="dash", line_color="red", annotation_text="Market Avg")
            fig_ret.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_ret, use_container_width=True)
            
        with bar_col2:
            fig_vol = px.bar(port, x='Ann_Volatility', y='Symbol', orientation='h', title='Volatility (lower = safer)', color_discrete_sequence=['steelblue'], height=350)
            fig_vol.add_vline(x=stats['Ann_Volatility'].mean(), line_dash="dash", line_color="red", annotation_text="Market Avg")
            fig_vol.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_vol, use_container_width=True)
            
        with bar_col3:
            fig_sh = px.bar(port, x='Sharpe_Ratio', y='Symbol', orientation='h', title='Sharpe Ratio', color_discrete_sequence=['mediumpurple'], height=350)
            fig_sh.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Sharpe=1.0")
            fig_sh.add_vline(x=stats['Sharpe_Ratio'].mean(), line_dash="dot", line_color="orange", annotation_text="Market Avg")
            fig_sh.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_sh, use_container_width=True)
            
        with bar_col4:
            fig_beta = px.bar(port, x='Beta', y='Symbol', orientation='h', title='Beta (lower = defensive)', color_discrete_sequence=['darkorange'], height=350)
            fig_beta.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Market Beta=1.0")
            fig_beta.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_beta, use_container_width=True)

        st.markdown("---")
        st.subheader("Portfolio Comparison — Radar Chart")

        radar_data = {name: [
            p['Ann_Return'].mean(),
            100 - p['Ann_Volatility'].mean(),
            p['Sharpe_Ratio'].mean() * 100,
            100 + p['Max_Drawdown'].mean()
        ] for name, p in portfolios.items()}

        all_vals     = [v for vals in radar_data.values() for v in vals]
        min_v, max_v = min(all_vals), max(all_vals)
        normalized   = {k: [(v - min_v) / (max_v - min_v) * 100 for v in vals]
                        for k, vals in radar_data.items()}

        labels   = ['Annual Return', 'Safety', 'Sharpe Ratio', 'Drawdown Protection']
        
        fig_radar = go.Figure()
        
        colors = ['mediumseagreen', 'steelblue', 'tomato']
        for (name, values), color in zip(normalized.items(), colors):
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill='toself',
                name=name,
                line_color=color
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title='Portfolio Comparison Radar Chart<br><sup>(Higher = Better on all axes)</sup>',
            height=600
        )
        st.plotly_chart(fig_radar, use_container_width=True)
