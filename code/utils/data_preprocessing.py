import pandas as pd
import ta
import yfinance as yf


def load_data(ticker="SPY"):
    """
    Loads SPY, VIX, and Interest Rates, then merges them into one dataframe.
    """
    print(f"Downloading {ticker}, ^VIX, and ^TNX...")

    # 1. Download Main Asset
    df = yf.download(ticker, start="2015-01-01", end="2024-12-31")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    # 2. Download VIX (Fear Index)
    vix = yf.download("^VIX", start="2015-01-01", end="2024-12-31")
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix.reset_index()[["Date", "Close"]].rename(columns={"Close": "VIX"})

    # 3. Download TNX (10-Year Treasury Yield)
    tnx = yf.download("^TNX", start="2015-01-01", end="2024-12-31")
    if isinstance(tnx.columns, pd.MultiIndex):
        tnx.columns = tnx.columns.get_level_values(0)
    tnx = tnx.reset_index()[["Date", "Close"]].rename(columns={"Close": "TNX"})

    # 4. Merge Data (Align dates)
    df = df.merge(vix, on="Date", how="inner")
    df = df.merge(tnx, on="Date", how="inner")
    df = df.sort_values("Date").reset_index(drop=True)

    # ==========================
    # Technical Indicators
    # ==========================

    # Existing Indicators
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()
    df["Returns"] = df["Close"].pct_change()

    # --- NEW FEATURES ---
    # 1. Average True Range (Volatility)
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()

    # 2. On-Balance Volume (Momentum)
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()

    # Drop NaNs created by rolling windows
    df = df.dropna().reset_index(drop=True)

    return df


def get_train_test_data(ticker="SPY", train_ratio=0.8):
    """
    Updated to include new columns in normalization.
    """
    df = load_data(ticker)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Add new features to the list
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "RSI", "MACD", "MACD_signal", "MACD_hist",
        "SMA_20", "SMA_50", "BB_high", "BB_low", "Returns",
        "VIX", "TNX", "ATR", "OBV"
    ]

    # Scale based on TRAIN only
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    for df_subset in [train_df, test_df]:
        scaled_data = scaler.transform(df_subset[feature_cols])
        for i, col in enumerate(feature_cols):
            df_subset[f"{col}_norm"] = scaled_data[:, i]

    return train_df, test_df