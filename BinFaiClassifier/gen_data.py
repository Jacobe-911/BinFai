import pandas as pd
import numpy as np
import os

# 配置参数
window_size = 20       # 输入序列长度
pred_horizon = 1       # 预测未来第几根K线涨跌
feature_cols = ['upper_shadow_ratio', 'lower_shadow_ratio', 'body_ratio', 'continuous_trend', 'RSI_6', 'EMA6_diff_ratio', 'ATR6_ratio']
label_col = 'target'   # 预测标签列
periods = ['10min', '30min', '60min', '1d']
data_dir = './data'    # 你存csv文件的目录
save_dir = './processed_data'  # 处理后保存目录

os.makedirs(save_dir, exist_ok=True)

def generate_target_labels(df, pred_horizon):
    df = df.copy()
    # 用 continuous_trend 的未来值生成涨跌标签
    # 未来第pred_horizon根的 continuous_trend >0 就标1，否则0
    df[label_col] = (df['continuous_trend'].shift(-pred_horizon) > 0).astype(int)
    return df.dropna(subset=[label_col])  # 删除因shift产生的空值行

def load_and_split_xy(df, feature_cols, target_col, window_size, pred_horizon):
    X, y = [], []
    for i in range(len(df) - window_size - pred_horizon + 1):
        x_window = df.iloc[i:i+window_size][feature_cols].values
        y_label = df.iloc[i+window_size+pred_horizon-1][target_col]
        X.append(x_window)
        y.append(y_label)
    return np.array(X), np.array(y)

def process_file(file_path):
    print(f"处理文件：{file_path}")
    df = pd.read_csv(file_path, parse_dates=['open_time'], index_col='open_time')
    df = generate_target_labels(df, pred_horizon)
    X, y = load_and_split_xy(df, feature_cols, label_col, window_size, pred_horizon)
    return X, y

def main():
    for period in periods:
        for dataset_type in ['train', 'test']:
            file_path = os.path.join(data_dir, f"{dataset_type}_{period}.csv")
            X, y = process_file(file_path)

            save_x_path = os.path.join(save_dir, f"{dataset_type}_{period}_X.npy")
            np.save(save_x_path, X)
            print(f"保存X到 {save_x_path}")

            save_y_path = os.path.join(save_dir, f"{dataset_type}_{period}_y.npy")
            np.save(save_y_path, y)
            print(f"保存y到 {save_y_path}")

if __name__ == "__main__":
    main()
