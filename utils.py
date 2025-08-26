def create_features(df, for_simulation=False):
    df = df.copy()
    
    # Lagged returns
    for col in ["Open","High","Low","Close"]:
        df[f'Return_{col}_1'] = (df[col] - df[col].shift(1)) / df[col].shift(1)
    
    # Rolling means
    df['MA_Close_5'] = df['Close'].rolling(5).mean()
    df['MA_Close_10'] = df['Close'].rolling(10).mean()
    
    # Rolling volatility
    df['Vol_Close_5'] = df['Close'].rolling(5).std()
    df['Vol_High_5'] = df['High'].rolling(5).std()
    
    # Rolling high/low
    df['Rolling_High_5'] = df['High'].rolling(5).max()
    df['Rolling_Low_5'] = df['Low'].rolling(5).min()
    
    # Normalized features
    df['HL_Ratio_5'] = (df['Rolling_High_5'] - df['Rolling_Low_5']) / df['Close']
    df['Close_vs_MA5'] = df['Close'] / df['MA_Close_5'] - 1
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    lag_cols = ['MA_Close_5','Vol_Close_5','HL_Ratio_5','Close_vs_MA5','RSI_14','MACD']
    for col in lag_cols:
        df[f'{col}_lag'] = df[col].shift(1)
    
    df['Target'] = (df['Close'].shift(-1) - df['Close']) / df['Close']
    
    if for_simulation:
        # Replace deprecated fillna(method='ffill')
        df = df.ffill().fillna(0)
    else:
        df = df.dropna().reset_index(drop=True)

    feature_cols = [f'Return_{col}_1' for col in ["Open","High","Low","Close"]] + \
                   ['MA_Close_5_lag','MA_Close_10','Vol_Close_5_lag','Vol_High_5',
                    'Rolling_High_5','Rolling_Low_5','HL_Ratio_5_lag','Close_vs_MA5_lag',
                    'RSI_14_lag','MACD_lag','Open','High','Low','Close','Volume']
    
    return df, feature_cols