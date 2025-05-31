from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import requests
import numpy as np
import uvicorn

app = FastAPI()


class PredictRequest(BaseModel):
    interval: str  # K线周期，比如 '1m', '5m', '30m' 等


# 你的模型定义，和之前一样（简化示例）
class CNN_BiLSTM_Attention(torch.nn.Module):
    def __init__(self, input_dim=8, cnn_out_channels=64, lstm_hidden_dim=64, num_classes=2):
        super().__init__()
        self.cnn = torch.nn.Conv1d(input_dim, cnn_out_channels, kernel_size=3, padding=1)
        self.bilstm = torch.nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_dim, batch_first=True,
                                    bidirectional=True)
        self.attention = torch.nn.Linear(lstm_hidden_dim * 2, 1)
        self.fc = torch.nn.Linear(lstm_hidden_dim * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc(context)
        return out


# 加载模型权重（请替换成你的路径）
model = CNN_BiLSTM_Attention()
model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
model.eval()

BINANCE_API_URL = 'https://api.binance.com/api/v3/klines'


def fetch_binance_klines(symbol='BTCUSDT', interval='30m', limit=20):
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(BINANCE_API_URL, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail='Failed to fetch Kline data from Binance')
    data = response.json()
    # 每条k线返回格式：
    # [OpenTime, Open, High, Low, Close, Volume, CloseTime, QuoteAssetVolume, NumberOfTrades, ...]
    return data


def preprocess_klines(raw_klines):
    # 这里你根据你的模型输入特征来转换数据
    # 假设你用的是8个特征：开盘价，最高价，最低价，收盘价，成交量，和3个计算指标（示例）
    processed = []
    for k in raw_klines:
        open_p = float(k[1])
        high_p = float(k[2])
        low_p = float(k[3])
        close_p = float(k[4])
        volume = float(k[5])
        # 示例计算三个简单指标（你换成自己特征计算逻辑）
        upper_shadow = high_p - max(open_p, close_p)
        lower_shadow = min(open_p, close_p) - low_p
        body = abs(close_p - open_p)
        features = [open_p, high_p, low_p, close_p, volume, upper_shadow, lower_shadow, body]
        processed.append(features)
    return np.array(processed, dtype=np.float32)


@app.post('/predict')
def predict(req: PredictRequest):
    # 1. 从Binance拉取最新20根K线
    raw_klines = fetch_binance_klines(interval=req.interval)
    # 2. 预处理成模型输入
    input_features = preprocess_klines(raw_klines)
    input_tensor = torch.tensor(input_features).unsqueeze(0)  # batch=1, seq_len=20, feat=8

    # 3. 模型预测
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0].tolist()
        pred_class = int(np.argmax(probs))

    return {
        "predicted_class": pred_class,
        "probabilities": probs,
        "used_interval": req.interval,
        "raw_kline_count": len(raw_klines)
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
