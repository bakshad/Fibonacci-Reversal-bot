
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from tqdm import tqdm

def get_nifty50_symbols():
    return [
        'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'ITC', 'HINDUNILVR',
        'LT', 'KOTAKBANK', 'AXISBANK', 'BAJFINANCE', 'BHARTIARTL', 'MARUTI', 'TITAN',
        'ASIANPAINT', 'SUNPHARMA', 'WIPRO', 'TECHM', 'NESTLEIND', 'ULTRACEMCO',
        'TATASTEEL', 'NTPC', 'POWERGRID', 'HCLTECH', 'INDUSINDBK', 'CIPLA',
        'GRASIM', 'BAJAJFINSV', 'COALINDIA', 'TATAMOTORS', 'JSWSTEEL', 'HINDALCO',
        'BPCL', 'ONGC', 'DRREDDY', 'BRITANNIA', 'DIVISLAB', 'EICHERMOT', 'BAJAJ_AUTO',
        'SBILIFE', 'HDFCLIFE', 'HEROMOTOCO', 'ADANIPORTS', 'APOLLOHOSP', 'UPL',
        'ICICIPRULI', 'TATACONSUM', 'SHREECEM'
    ]

START_DATE = (datetime.today() - timedelta(days=180)).strftime('%Y-%m-%d')
END_DATE = datetime.today().strftime('%Y-%m-%d')

def compute_indicators(df):
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df['MACD_DIFF'] = MACD(df['Close']).macd_diff()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['Prev_Change'] = df['Close'].pct_change().shift(1) * 100
    return df

def detect_reversals(df, sym):
    df['Target'] = np.nan
    df['SignalType'] = np.nan
    df['Option'] = np.nan
    df['SL'] = np.nan
    df['TargetPrice'] = np.nan

    for i in range(1, len(df)):
        prev_close = df.loc[i-1, 'Close']
        curr_close = df.loc[i, 'Close']
        prev_rsi = df.loc[i-1, 'RSI']
        curr_rsi = df.loc[i, 'RSI']
        lower = df.loc[i-1, 'BB_lower']
        upper = df.loc[i-1, 'BB_upper']
        atr = df.loc[i, 'ATR']

        if np.isnan([prev_close, curr_close, prev_rsi, curr_rsi, lower, upper, atr]).any():
            continue

        if prev_close < lower and curr_close > prev_close and curr_rsi > prev_rsi:
            strike = int(round(curr_close / 50.0) * 50 + 50)
            df.at[i, 'Target'] = 1
            df.at[i, 'SignalType'] = 'Bullish'
            df.at[i, 'Option'] = f"{sym} {strike} CE"
            df.at[i, 'SL'] = round(curr_close - 0.3 * atr, 2)
            df.at[i, 'TargetPrice'] = round(curr_close + 0.9 * atr, 2)

        elif prev_close > upper and curr_close < prev_close and curr_rsi < prev_rsi:
            strike = int(round(curr_close / 50.0) * 50 - 50)
            df.at[i, 'Target'] = 0
            df.at[i, 'SignalType'] = 'Bearish'
            df.at[i, 'Option'] = f"{sym} {strike} PE"
            df.at[i, 'SL'] = round(curr_close + 0.3 * atr, 2)
            df.at[i, 'TargetPrice'] = round(curr_close - 0.9 * atr, 2)

    df['Symbol'] = sym
    return df.dropna(subset=['Target'])

def build_dataset(symbols):
    all_data = []
    for sym in tqdm(symbols):
        try:
            df = yf.download(f"{sym}.NS", start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
            if df.empty or len(df) < 30:
                continue
            df = df.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = compute_indicators(df)
            df = detect_reversals(df, sym)
            if not df.empty:
                df['RSI_DIFF'] = df['RSI'] - df['RSI'].shift(1)
                all_data.append(df)
        except Exception as e:
            continue
    return pd.concat(all_data) if all_data else pd.DataFrame()

if __name__ == "__main__":
    print("📈 Generating Bollinger Band Reversal signals for Nifty 50...")
    symbols = get_nifty50_symbols()
    df = build_dataset(symbols)
    if not df.empty:
        df.sort_values(by='RSI_DIFF', ascending=False, inplace=True)
        top_df = df.tail(5).sort_values(by='Date')
        top_df.to_csv("bollinger_reversal_signals.csv", index=False)
        print(f"✅ Saved top {len(top_df)} signals to bollinger_reversal_signals.csv")
    else:
        empty = pd.DataFrame(columns=[
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'BB_upper', 'BB_lower',
            'RSI', 'MACD_DIFF', 'ATR', 'Prev_Change', 'Target', 'SignalType', 'Option',
            'SL', 'TargetPrice', 'Symbol', 'RSI_DIFF'
        ])
        empty.to_csv("bollinger_reversal_signals.csv", index=False)
        print("❌ No signals detected. Saved empty CSV.")
