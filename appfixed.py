import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

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
            import requests
            file_id = "1YXM-hBa_2orAI2eMyyE5XE1Qrs4kJtVU"
            # Step 1 — start download session
            session = requests.Session()
            url = "https://drive.google.com/uc?export=download"
            response = session.get(url, params={'id': file_id}, stream=True)
            # Step 2 — handle large file confirmation token
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            if token:
                response = session.get(url, params={'id': file_id, 'confirm': token}, stream=True)
            # Step 3 — save file
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(32768):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            st.error(f"❌ Could not download dataset: {e}")
            return None
    df = pd.read_csv(local_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    return df
    

# ── LOAD MARKET DATA (from CSV, no yfinance needed) ───
@st.cache_data
def load_market(df_json):
    df = pd.read_json(df_json)
    df['Date'] = pd.to_datetime(df['Date'])
    # ✅ Compute market return as equal-weighted average of all 50 stocks
    daily = (
        df.groupby(['Symbol', 'Date'])['Close']
        .last()
        .groupby('Date')
        .mean()
        .pct_change() * 100
    )
    market = daily.reset_index()
    market.columns = ['Date', 'Market_Return']
    return market.dropna().reset_index(drop=True)

# ── COMPUTE STATS ──────────────────────────────────────
@st.cache_data
def compute_stats(df_json, market_json):
    df     = pd.read_json(df_json)
    market = pd.read_json(market_json)
    df['Date']     = pd.to_datetime(df['Date'])
    market['Date'] = pd.to_datetime(market['Date'])
    market = market.set_index('Date')

    results = []
    for symbol in df['Symbol'].unique():
        try:
            s = df[df['Symbol'] == symbol].copy()
            s['Daily_Return'] = s['Close'].pct_change()
            s = s.dropna(subset=['Daily_Return'])
            s = s.set_index('Date')

            ann_return  = s['Daily_Return'].mean() * 252 * 100
            ann_vol     = s['Daily_Return'].std() * np.sqrt(252) * 100
            sharpe      = (ann_return/100) / (ann_vol/100) if ann_vol != 0 else 0
            rolling_max = s['Close'].cummax()
            max_dd      = ((s['Close'] - rolling_max) / rolling_max).min() * 100

            combined = s[['Daily_Return']].copy()
            combined.columns = ['Stock_Return']
            combined['Stock_Return'] *= 100
            combined = combined.join(market[['Market_Return']], how='inner').dropna()

            beta = 0
            corr = 0
            if len(combined) > 100:
                cov  = np.cov(combined['Stock_Return'], combined['Market_Return'])
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0
                corr = combined['Stock_Return'].corr(combined['Market_Return'])

            results.append({
                'Symbol'        : symbol,
                'Ann_Return'    : round(ann_return, 2),
                'Ann_Volatility': round(ann_vol, 2),
                'Sharpe_Ratio'  : round(sharpe, 3),
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
    st.error("❌ NIFTY50_all.csv not found! Please upload it to Colab first.")
    st.code("from google.colab import files\nfiles.upload()  # upload NIFTY50_all.csv")
    st.stop()

market = load_market(df.to_json())
stats  = compute_stats(df.to_json(), market.to_json())
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

portfolios = {
    'Conservative': conservative,
    'Balanced'    : balanced,
    'Aggressive'  : aggressive
}

# ── SIDEBAR ────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/NSE_logo.svg/320px-NSE_logo.svg.png",
    width=200
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
    st.markdown("### What this platform does")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**📊 Stock Analyzer**\nAnalyze any NIFTY-50 stock — price trends, moving averages, returns, volatility and ML-based price prediction")
        st.success("**⚠️ Risk & Beta Dashboard**\nQuantify risk using Sharpe Ratio, Max Drawdown, Volatility and Beta sensitivity analysis")
    with col2:
        st.warning("**💼 Portfolio Constructor**\nBuild optimized portfolios for Conservative, Balanced and Aggressive investor profiles")
        st.error("**🔍 Explainable Recommendations**\nUnderstand exactly WHY each stock was selected using 4-metric visual justification")

    st.markdown("---")
    st.markdown("### Dataset Overview")
    st.dataframe(df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']].head(10))

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
    col4.metric("Avg Daily Return", f"{stock['Daily_Return'].mean():.3f}%")

    st.markdown("---")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(stock['Date'], stock['Close'],    color='steelblue', linewidth=0.8, label='Close Price', alpha=0.7)
    axes[0].plot(stock['Date'], stock['MA20'],     color='orange',    linewidth=1.5, label='MA20')
    axes[0].plot(stock['Date'], stock['MA50'],     color='red',       linewidth=1.5, label='MA50')
    axes[0].set_title(f'{symbol} — Price with Moving Averages')
    axes[0].set_ylabel('Price (₹)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(stock['Date'], stock['Volatility'], color='red', linewidth=1)
    axes[1].set_title(f'{symbol} — 30-Day Rolling Volatility')
    axes[1].set_ylabel('Volatility')
    axes[1].set_xlabel('Year')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ML Prediction
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

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE",  f"₹{mae:.2f}")
        col2.metric("RMSE", f"₹{rmse:.2f}")
        col3.metric("R²",   f"{r2:.4f}")

        fig2, ax = plt.subplots(figsize=(14, 4))
        ax.plot(y_test.values, color='steelblue', linewidth=1,   label='Actual Price')
        ax.plot(y_pred,        color='orange',    linewidth=1,   label='Predicted Price', alpha=0.8)
        ax.set_title(f'{symbol} — Actual vs Predicted Price')
        ax.set_xlabel('Trading Days (Test Set)')
        ax.set_ylabel('Price (₹)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.markdown("---")
        st.subheader("🔮 Next Day Prediction")
        last_row   = stock_clean[features].iloc[-1]
        next_price = model.predict([last_row])[0]
        last_price = stock_clean['Close'].iloc[-1]
        change     = next_price - last_price
        pct_change = (change / last_price) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Last Known Price",     f"₹{last_price:.2f}")
        col2.metric("Predicted Next Price", f"₹{next_price:.2f}", f"{change:+.2f}")
        col3.metric("Expected Change",      f"{pct_change:+.2f}%")
        st.caption("⚠️ This prediction is based on historical patterns only. Not financial advice.")

# ══════════════════════════════════════════════════════
# PAGE 3 — PORTFOLIO CONSTRUCTOR
# ══════════════════════════════════════════════════════
elif page == "💼 Portfolio Constructor":
    st.title("💼 Portfolio Constructor")

    col1, col2, col3 = st.columns(3)
    for col, (name, port), color in zip(
        [col1, col2, col3],
        portfolios.items(),
        ['🟢', '🔵', '🔴']
    ):
        with col:
            st.markdown(f"### {color} {name}")
            st.metric("Avg Annual Return", f"{port['Ann_Return'].mean():.1f}%")
            st.metric("Avg Volatility",    f"{port['Ann_Volatility'].mean():.1f}%")
            st.metric("Avg Sharpe Ratio",  f"{port['Sharpe_Ratio'].mean():.3f}")
            st.metric("Avg Max Drawdown",  f"{port['Max_Drawdown'].mean():.1f}%")
            st.metric("Avg Beta",          f"{port['Beta'].mean():.3f}")

    st.markdown("---")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, port) in zip(axes, portfolios.items()):
        weights = [100 / len(port)] * len(port)
        ax.pie(weights, labels=port['Symbol'].values,
               autopct='%1.0f%%', startangle=90,
               colors=plt.cm.Set3.colors[:len(port)],
               textprops={'fontsize': 8})
        ax.set_title(f'{name}\nReturn: {port["Ann_Return"].mean():.1f}% | '
                     f'Risk: {port["Ann_Volatility"].mean():.1f}% | '
                     f'Sharpe: {port["Sharpe_Ratio"].mean():.2f}',
                     fontsize=10, fontweight='bold')
    plt.suptitle('Portfolio Allocations by Investor Profile', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    selected = st.selectbox("View detailed stock list", list(portfolios.keys()))
    st.dataframe(
        portfolios[selected][['Symbol', 'Ann_Return', 'Ann_Volatility', 'Sharpe_Ratio', 'Max_Drawdown', 'Beta']]
        .reset_index(drop=True)
        .style.format({
            'Ann_Return'    : '{:.2f}%',
            'Ann_Volatility': '{:.2f}%',
            'Sharpe_Ratio'  : '{:.3f}',
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

    col1, col2, col3, col4 = st.columns(4)
    safest   = stats.loc[stats['Ann_Volatility'].idxmin()]
    riskiest = stats.loc[stats['Ann_Volatility'].idxmax()]
    best_sh  = stats.loc[stats['Sharpe_Ratio'].idxmax()]
    worst_dd = stats.loc[stats['Max_Drawdown'].idxmin()]

    col1.metric("Safest Stock",   safest['Symbol'],   f"Vol: {safest['Ann_Volatility']:.1f}%")
    col2.metric("Riskiest Stock", riskiest['Symbol'], f"Vol: {riskiest['Ann_Volatility']:.1f}%")
    col3.metric("Best Sharpe",    best_sh['Symbol'],  f"Sharpe: {best_sh['Sharpe_Ratio']:.3f}")
    col4.metric("Worst Drawdown", worst_dd['Symbol'], f"DD: {worst_dd['Max_Drawdown']:.1f}%")

    st.markdown("---")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    beta_sorted  = stats.sort_values('Beta')
    colors_beta  = ['tomato' if b > 1.2 else 'steelblue' if b > 0.8 else 'mediumseagreen'
                    for b in beta_sorted['Beta']]
    axes[0, 0].barh(beta_sorted['Symbol'], beta_sorted['Beta'], color=colors_beta)
    axes[0, 0].axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, label='Beta=1.0')
    axes[0, 0].set_title('Beta — All Stocks\n(Green=Defensive, Blue=Moderate, Red=Aggressive)')
    axes[0, 0].set_xlabel('Beta')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    sc = axes[0, 1].scatter(stats['Ann_Volatility'], stats['Ann_Return'],
                            c=stats['Sharpe_Ratio'], cmap='RdYlGn',
                            s=80, alpha=0.8, edgecolors='white')
    plt.colorbar(sc, ax=axes[0, 1], label='Sharpe Ratio')
    for _, row in stats.iterrows():
        axes[0, 1].annotate(row['Symbol'], (row['Ann_Volatility'], row['Ann_Return']),
                            fontsize=6, alpha=0.7, xytext=(3, 3), textcoords='offset points')
    axes[0, 1].set_title('Risk vs Return (color = Sharpe Ratio)')
    axes[0, 1].set_xlabel('Volatility (Risk) %')
    axes[0, 1].set_ylabel('Annual Return %')
    axes[0, 1].grid(True, alpha=0.2)

    port_betas  = [stats[stats['Symbol'].isin(p['Symbol'])]['Beta'].mean()
                   for p in portfolios.values()]
    port_names  = list(portfolios.keys())
    port_colors = ['mediumseagreen', 'steelblue', 'tomato']
    bars = axes[1, 0].bar(port_names, port_betas, color=port_colors, alpha=0.8, edgecolor='white')
    axes[1, 0].axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Market Beta=1.0')
    for bar, val in zip(bars, port_betas):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    axes[1, 0].set_title('Average Portfolio Beta')
    axes[1, 0].set_ylabel('Beta')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    market_changes = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    for (name, _), beta, color in zip(portfolios.items(), port_betas, port_colors):
        sim = [m * beta for m in market_changes]
        axes[1, 1].plot(market_changes, sim, 'o-', linewidth=2,
                        label=f'{name} (β={beta:.2f})', color=color)
    axes[1, 1].plot(market_changes, market_changes, 'k--', linewidth=1, alpha=0.5, label='Market')
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='black', linewidth=0.5)
    axes[1, 1].set_title('Sensitivity Simulation')
    axes[1, 1].set_xlabel('Market Change (%)')
    axes[1, 1].set_ylabel('Portfolio Change (%)')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Risk & Beta Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════
# PAGE 5 — EXPLAINABLE RECOMMENDATIONS
# ══════════════════════════════════════════════════════
elif page == "🔍 Explainable Recommendations":
    st.title("🔍 Explainable Recommendations")
    st.markdown("Understand exactly **why** each stock was selected for each portfolio.")

    selected = st.selectbox("Select Portfolio", list(portfolios.keys()))
    port     = portfolios[selected]

    st.markdown("---")
    st.subheader(f"Why each stock is in the {selected} Portfolio")

    for _, row in port.iterrows():
        beta_val = stats[stats['Symbol'] == row['Symbol']]['Beta'].values
        beta_str = f"{beta_val[0]:.3f}" if len(beta_val) > 0 else "N/A"
        with st.expander(f"📌 {row['Symbol']} — Why selected?"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Annual Return", f"{row['Ann_Return']:.1f}%")
            col2.metric("Volatility",    f"{row['Ann_Volatility']:.1f}%")
            col3.metric("Sharpe Ratio",  f"{row['Sharpe_Ratio']:.3f}")
            col4.metric("Beta",          beta_str)

    st.markdown("---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    syms      = port['Symbol'].values
    beta_vals = [stats[stats['Symbol'] == s]['Beta'].values[0]
                 if len(stats[stats['Symbol'] == s]) > 0 else 0
                 for s in syms]

    axes[0, 0].barh(syms, port['Ann_Return'].values, color='mediumseagreen', alpha=0.8)
    axes[0, 0].axvline(x=stats['Ann_Return'].mean(), color='red', linestyle='--', label='Avg')
    axes[0, 0].set_title('Annual Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].barh(syms, port['Ann_Volatility'].values, color='steelblue', alpha=0.8)
    axes[0, 1].axvline(x=stats['Ann_Volatility'].mean(), color='red', linestyle='--', label='Avg')
    axes[0, 1].set_title('Volatility (lower = safer)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].barh(syms, port['Sharpe_Ratio'].values, color='mediumpurple', alpha=0.8)
    axes[1, 0].axvline(x=1.0, color='red', linestyle='--', label='Sharpe=1.0')
    axes[1, 0].axvline(x=stats['Sharpe_Ratio'].mean(), color='orange', linestyle=':', label='Avg')
    axes[1, 0].set_title('Sharpe Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].barh(syms, beta_vals, color='darkorange', alpha=0.8)
    axes[1, 1].axvline(x=1.0, color='red', linestyle='--', label='Market Beta=1.0')
    axes[1, 1].set_title('Beta (lower = more defensive)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Explainable Recommendations — {selected} Portfolio',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Radar chart
    st.markdown("---")
    st.subheader("Portfolio Comparison — Radar Chart")
    radar_data = {}
    for name, p in portfolios.items():
        radar_data[name] = [
            p['Ann_Return'].mean(),
            100 - p['Ann_Volatility'].mean(),
            p['Sharpe_Ratio'].mean() * 100,
            100 + p['Max_Drawdown'].mean()
        ]

    all_vals       = [v for vals in radar_data.values() for v in vals]
    min_v, max_v   = min(all_vals), max(all_vals)
    normalized     = {k: [(v - min_v) / (max_v - min_v) * 100 for v in vals]
                      for k, vals in radar_data.items()}

    labels   = ['Annual\nReturn', 'Safety', 'Sharpe\nRatio', 'Drawdown\nProtection']
    num_vars = len(labels)
    angles   = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles  += angles[:1]

    fig2, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors   = ['mediumseagreen', 'steelblue', 'tomato']
    for (name, values), color in zip(normalized.items(), colors):
        vals = values + values[:1]
        ax.plot(angles, vals, 'o-', linewidth=2.5, label=name, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_title('Portfolio Comparison Radar Chart\n(Higher = Better on all axes)',
                 size=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
