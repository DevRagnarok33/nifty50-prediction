# NIFTY-50 Investment Intelligence Platform

An advanced, data-driven financial dashboard designed to assist users in making informed investment decisions using Machine Learning and quantitative risk assessment.

## Key Features
- **Stock Analyzer:** Predicts future price trends using Linear Regression with multi-feature engineering (Lag features, Moving Averages, Volatility).
- **Portfolio Constructor:** Automatically generates Conservative, Balanced, and Aggressive portfolios based on Sharpe Ratio and historical volatility.
- **Risk & Beta Dashboard:** Analyzes market sensitivity (Beta), risk-adjusted returns (Sharpe/Sortino), and worst-case scenarios (Max Drawdown).
- **Explainable Recommendations:** Provides transparent, data-backed justification for every portfolio selection, comparing metrics against market averages.

## Installation & Setup
To run this application locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone [https://github.com/DevRagnarok33/nifty50-prediction/tree/main]

2. Install the required dependencies:
   bash
   pip install -r requirements.txt

3. Run the application:
   bash 
   streamlit run appfixed.py

Model Architecture
The platform utilizes dynamic model training. Rather than relying on static, pre-trained model files, the application trains individual LinearRegression models on-the-fly for the selected stock to ensure predictions remain sensitive to the most recent historical data trends.   
