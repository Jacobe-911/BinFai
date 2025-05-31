import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler

def resample_klines(df_5m, target_minutes):
    # 对齐索引到整点0分开始，方便分段
    # 先重采样前把索引向下取整到最近的target_minutes倍数分钟
    idx_floor = df_5m.index.floor(f'{target_minutes}T')
    df_5m = df_5m.copy()
    df_5m.index = idx_floor

    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_resampled = df_5m.resample(f'{target_minutes}T').agg(ohlc_dict).dropna()
    return df_resampled

def calc_continuous_trend(close_series):
    trend = [0]
    for i in range(1, len(close_series)):
        if close_series.iloc[i] > close_series.iloc[i - 1]:
            trend.append(trend[-1] + 1 if trend[-1] > 0 else 1)
        elif close_series.iloc[i] < close_series.iloc[i - 1]:
            trend.append(trend[-1] - 1 if trend[-1] < 0 else -1)
        else:
            trend.append(0)
    return pd.Series(trend, index=close_series.index)

def extract_features(df):
    df = df.copy()

    # 计算影线和实体长度
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body'] = (df['close'] - df['open']).abs()

    candle_range = df['high'] - df['low']
    candle_range = candle_range.replace(0, np.nan)  # 避免除零

    df['upper_shadow_ratio'] = df['upper_shadow'] / candle_range
    df['lower_shadow_ratio'] = df['lower_shadow'] / candle_range
    df['body_ratio'] = df['body'] / candle_range

    df['continuous_trend'] = calc_continuous_trend(df['close'])

    df['RSI_6'] = ta.momentum.RSIIndicator(df['close'], window=6).rsi()

    ema6 = ta.trend.EMAIndicator(df['close'], window=6).ema_indicator()
    df['EMA6_diff_ratio'] = (df['close'] - ema6) / df['close']

    atr6 = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=6).average_true_range()
    df['ATR6_ratio'] = atr6 / df['close']

    features = df[['upper_shadow_ratio', 'lower_shadow_ratio', 'body_ratio',
                   'continuous_trend', 'RSI_6', 'EMA6_diff_ratio', 'ATR6_ratio']]

    features = features.dropna()
    return features

def train_test_split_by_time(features, train_ratio=0.8):
    split_idx = int(len(features) * train_ratio)
    train = features.iloc[:split_idx]
    test = features.iloc[split_idx:]
    return train, test

def standardize_features(train, test):
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train), index=train.index, columns=train.columns)
    test_scaled = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
    return train_scaled, test_scaled, scaler

def process_period(df_5m, period_minutes):
    print(f"Processing period: {period_minutes} minutes")
    df_period = resample_klines(df_5m, period_minutes)
    features = extract_features(df_period)
    train, test = train_test_split_by_time(features)
    train_scaled, test_scaled, scaler = standardize_features(train, test)
    return train_scaled, test_scaled

def main(df_5m):
    # 确保索引是DatetimeIndex，且已排序
    df_5m = df_5m.sort_index()
    if not isinstance(df_5m.index, pd.DatetimeIndex):
        raise ValueError("df_5m index must be a DatetimeIndex")

    # 对齐起点到第一个整天0点0分
    start_date = df_5m.index[0].normalize()  # 当天0点
    df_5m = df_5m.loc[df_5m.index >= start_date]

    periods = [10, 30, 60, 1440]  # 单位分钟，1440=1天
    results = {}

    for p in periods:
        train_scaled, test_scaled = process_period(df_5m, p)
        results[p] = (train_scaled, test_scaled)

    return results

def save_datasets(results, folder='data'):
    import os
    os.makedirs(folder, exist_ok=True)

    for period, (train_df, test_df) in results.items():
        train_path = f"{folder}/train_{period}min.csv"
        test_path = f"{folder}/test_{period}min.csv"
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
        print(f"Saved train to {train_path}, test to {test_path}")

if __name__ == '__main__':
    # 你这里读入5分钟k线CSV，或者其它方式拿到DataFrame df_5m
    # 示例：
    df_5m = pd.read_csv('data/binance_BTCUSDT_5m_raw.csv', index_col=0, parse_dates=True)
    df_5m = df_5m[['open', 'high', 'low', 'close', 'volume']].astype(float).copy()

    results = main(df_5m)
    save_datasets(results)

    for period, (train, test) in results.items():
        print(f"Period: {period} min, train shape: {train.shape}, test shape: {test.shape}")

