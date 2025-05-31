import requests
import pandas as pd
from datetime import datetime, timedelta
import time


# ---------- 下载币安5分钟K线 ----------

def fetch_binance_klines(symbol='BTCUSDT', interval='5m', start_str=None, end_str=None):
    """
    从币安获取K线，支持分页批量拉取
    返回 DataFrame，列包括：
    ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
     'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    """

    limit = 1000  # 单次最大数据条数
    base_url = 'https://api.binance.com/api/v3/klines'

    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000) if start_str else None
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else None

    all_klines = []
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
        }
        if start_ts:
            params['startTime'] = start_ts
        if end_ts:
            params['endTime'] = end_ts

        resp = requests.get(base_url, params=params)
        data = resp.json()
        if not data:
            break

        all_klines.extend(data)
        last_open_time = data[-1][0]

        # 下一次起点，避免重复
        start_ts = last_open_time + 1

        if len(data) < limit:
            break

        # 避免请求频率过快
        time.sleep(0.5)

    # 转换成DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    # 数据类型转换
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)

    return df


def main():
    symbol = 'BTCUSDT'
    start_date = (datetime.utcnow() - timedelta(days=365 * 5)).strftime('%Y-%m-%d %H:%M:%S')
    end_date = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    print(f"下载5分钟K线数据：{start_date} ~ {end_date}")
    df_5m = fetch_binance_klines(symbol=symbol, interval='5m', start_str=start_date, end_str=end_date)
    # 直接保存成CSV
    df_5m.to_csv('data/binance_BTCUSDT_5m_raw.csv')
    # 如果你想保存成更高效的二进制格式也可以用pickle
    df_5m.to_pickle('data/binance_BTCUSDT_5m_raw.pkl')
    print(f"5分钟K线数量: {len(df_5m)}")


if __name__ == "__main__":
    main()
