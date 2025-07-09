
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from tqdm import tqdm

# Hardcoded fallback list
def get_fno_symbols():
    return ['RELIANCE', 'TCS', 'INFY', 'ICICIBANK', 'HDFCBANK', 'SBIN', 'HINDUNILVR', 'ITC']

START_DATE = (datetime.today() - timedelta(days=180)).strftime('%Y-%m-%d')  # 180-day lookback
END_DATE = datetime.today().strftime('%Y-%m-%d')

def generate_features(df):
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    macd = MACD(df['Close'])
    df['MACD_DIFF'] = macd.macd_diff()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['Prev_Change'] = df['Close'].pct_change().shift(1) * 100
    return df

def label_target(df):
    df['Target'] = 0
    for i in range(len(df) - 4):
        atr = df.loc[i, 'ATR']
        entry_price = df.loc[i, 'Close']
        if np.isnan(atr) or atr == 0:
            continue
        future_high = df.loc[i+1:i+3, 'High'].max()
        if (future_high - entry_price) >= 1.25 * atr:
            df.at[i, 'Target'] = 1
    return df

def build_training_data(symbols):
    all_data = []
    good_symbols = []
    for sym in tqdm(symbols[:100]):
        try:
            df = yf.download(f"{sym}.NS", start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
            if df.empty or len(df) < 30:
                continue
            df.reset_index(inplace=True)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = generate_features(df)
            df = label_target(df)
            df['Symbol'] = sym
            cols = ['Date', 'Symbol', 'RSI', 'MACD_DIFF', 'Volume', 'ATR', 'Prev_Change', 'Target']
            df_filtered = df[cols]
            df_filtered = df_filtered[df_filtered['Target'].notna()]
            if len(df_filtered) >= 10:  # lowered threshold to 10
                all_data.append(df_filtered)
                good_symbols.append(f"{sym} ({len(df_filtered)} rows)")
        except Exception as e:
            continue
    if not all_data:
        print("⚠️ No usable data collected. Try increasing lookback or checking data quality.")
        return pd.DataFrame()
    print(f"✅ Final symbols used with rows:\n - " + "\n - ".join(good_symbols))
    return pd.concat(all_data)

if __name__ == "__main__":
    print("Fetching F&O symbols...")
    symbols = get_fno_symbols()
    print(f"Collected {len(symbols)} symbols. Generating features and labels...")
    df = build_training_data(symbols)
    if not df.empty:
        df.to_csv('reversal_training_data.csv', index=False)
        print("✅ Saved to reversal_training_data.csv")
    else:
        print("❌ No data saved due to empty dataset.")
