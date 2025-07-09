
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from tqdm import tqdm

def get_fno_symbols():
    url = 'https://archives.nseindia.com/content/fo/fo_mktlots.csv'
    try:
        df = pd.read_csv(url)
        return df['SYMBOL'].dropna().unique().tolist()
    except:
        return ['RELIANCE', 'TCS', 'INFY']

START_DATE = (datetime.today() - timedelta(days=180)).strftime('%Y-%m-%d')
END_DATE = datetime.today().strftime('%Y-%m-%d')

def compute_indicators(df):
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_bandwidth'] = df['BB_upper'] - df['BB_lower']

    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['MACD_DIFF'] = MACD(df['Close']).macd_diff()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['Prev_Change'] = df['Close'].pct_change().shift(1) * 100

    return df

def detect_bollinger_reversals(df, sym):
    df['Target'] = np.nan
    df['SignalType'] = np.nan

    for i in range(1, len(df)):
        prev_close = df.loc[i-1, 'Close']
        curr_close = df.loc[i, 'Close']
        prev_rsi = df.loc[i-1, 'RSI']
        curr_rsi = df.loc[i, 'RSI']
        lower = df.loc[i-1, 'BB_lower']
        upper = df.loc[i-1, 'BB_upper']

        # Bullish reversal
        if prev_close < lower and curr_close > prev_close and curr_rsi > prev_rsi:
            df.at[i, 'Target'] = 1
            df.at[i, 'SignalType'] = 'Bullish'

        # Bearish reversal
        elif prev_close > upper and curr_close < prev_close and curr_rsi < prev_rsi:
            df.at[i, 'Target'] = 0
            df.at[i, 'SignalType'] = 'Bearish'

    df['Symbol'] = sym
    return df.dropna(subset=['Target'])

def build_bollinger_dataset(symbols):
    final = []
    for sym in tqdm(symbols):
        try:
            df = yf.download(f"{sym}.NS", start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
            if len(df) < 30:
                continue
            df = df.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = compute_indicators(df)
            df = detect_bollinger_reversals(df, sym)
            if not df.empty:
                final.append(df)
        except Exception as e:
            continue
    return pd.concat(final) if final else pd.DataFrame()

if __name__ == "__main__":
    print("⏳ Fetching real Bollinger Band reversals...")
    symbols = get_fno_symbols()
    df = build_bollinger_dataset(symbols)
    if not df.empty:
        df.to_csv("bollinger_reversal_data.csv", index=False)
        print(f"✅ Saved {len(df)} live signals to bollinger_reversal_data.csv")
    else:
        print("❌ No Bollinger Band reversal signals found.")
