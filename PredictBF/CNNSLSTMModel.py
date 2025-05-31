import torch
import torch.nn as nn


class CNNSLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, cnn_out_channels=32, lstm_layers=1, dropout=0.3):
        super(CNNSLSTMModel, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch_size, time_steps, input_dim)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, time_steps) -> for CNN
        x = self.relu(self.cnn(x))  # (batch, cnn_out_channels, time_steps)
        x = x.permute(0, 2, 1)  # (batch_size, time_steps, cnn_out_channels) -> for LSTM

        output, _ = self.lstm(x)  # (batch_size, time_steps, hidden_dim)
        x = output[:, -1, :]  # 取最后一个时间步的输出
        x = self.dropout(x)
        x = self.fc(x)
        prob = torch.sigmoid(x)  # 输出上涨概率
        return prob
