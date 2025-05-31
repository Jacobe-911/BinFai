import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# -------- Dataset定义 --------
class KLineDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, T, F)
        self.y = torch.tensor(y, dtype=torch.long)     # 改成long，用于CrossEntropyLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------- 模型定义 --------
class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, cnn_channels=64, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super(CNN_BiLSTM_Attention, self).__init__()
        # 1D CNN提取局部特征
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU()
        )

        # 双向LSTM捕捉时序依赖
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

        # 注意力层参数
        self.attention_w = nn.Linear(lstm_hidden * 2, 64)
        self.attention_u = nn.Linear(64, 1, bias=False)

        # 分类输出层
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # 输出两个类别logits
        )

    def attention(self, lstm_outputs):
        # lstm_outputs: (batch, seq_len, hidden*2)
        attn_scores = torch.tanh(self.attention_w(lstm_outputs))  # (batch, seq_len, 64)
        attn_scores = self.attention_u(attn_scores).squeeze(-1)  # (batch, seq_len)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len)
        attn_weights = attn_weights.unsqueeze(-1)  # (batch, seq_len, 1)
        weighted_output = torch.sum(lstm_outputs * attn_weights, dim=1)  # (batch, hidden*2)
        return weighted_output

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        x = x.permute(0, 2, 1)  # CNN expects (batch, channels, seq_len)
        cnn_out = self.cnn(x)  # (batch, cnn_channels, seq_len)
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, seq_len, cnn_channels)

        lstm_out, _ = self.lstm(cnn_out)  # (batch, seq_len, lstm_hidden*2)

        attn_out = self.attention(lstm_out)  # (batch, lstm_hidden*2)

        logits = self.fc(attn_out)  # (batch, 2)
        return logits

# -------- 计算准确率 --------
def calc_accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

# -------- 训练函数 --------
def train(model, device, train_loader, val_loader, epochs, criterion, optimizer, save_path):
    best_val_loss = float('inf')
    patience = 5
    trigger_times = 0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            total_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            total_samples += X_batch.size(0)

        avg_train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)
                val_correct += (outputs.argmax(dim=1) == y_val).sum().item()
                val_samples += X_val.size(0)
        avg_val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples

        print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 早停判断
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            trigger_times = 0
            print(f'Saved best model at epoch {epoch}')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping triggered.')
                break

# -------- 测试函数 --------
def test(model, device, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_batch.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = (y_true == y_pred).mean()
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# -------- 主入口 --------
def main():
    # 参数
    batch_size = 64
    epochs = 50
    lr = 1e-3
    model_path = 'model/model_60min.pth'

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    X_train = np.load('processed_data/train_60min_X.npy')  # (N, T, F)
    y_train = np.load('processed_data/train_60min_y.npy')
    X_test = np.load('processed_data/test_60min_X.npy')
    y_test = np.load('processed_data/test_60min_y.npy')

    # 简单划分训练集和验证集
    val_ratio = 0.1
    val_size = int(len(X_train) * val_ratio)
    train_size = len(X_train) - val_size

    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]

    # 创建DataLoader
    train_dataset = KLineDataset(X_train, y_train)
    val_dataset = KLineDataset(X_val, y_val)
    test_dataset = KLineDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 构建模型
    feature_dim = X_train.shape[2]
    model = CNN_BiLSTM_Attention(feature_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    train(model, device, train_loader, val_loader, epochs, criterion, optimizer, model_path)

    # 加载最佳模型
    model.load_state_dict(torch.load(model_path))

    # 测试
    test(model, device, test_loader)

if __name__ == '__main__':
    main()
