# 実際にNNで分類してみる
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselib import *

# モデル定義。だいぶ適当
class TrackerClassifier(nn.Module):
    def __init__(self, input_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), # 安定化
            nn.ReLU(),
            nn.Dropout(0.5),      # 強めのDropout
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# データ準備
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_pickle("js_vector_dataset.pkl")

X = np.stack(df['vector'].values)
y = df['label'].values

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.FloatTensor(X_train_raw)
y_train = torch.FloatTensor(y_train_raw).view(-1, 1)
X_test = torch.FloatTensor(X_test_raw)
y_test = torch.FloatTensor(y_test_raw).view(-1, 1)

classifier = TrackerClassifier().to(device)


# 学習設定
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)
epochs = 100

# 学習ループ
classifier.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = classifier(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# テストデータで予測
classifier.eval()
with torch.no_grad():
    y_prob = classifier(X_test.to(device)).cpu()
    y_pred = (y_prob > 0.9).float()

    # 真陽性(TP), 真陰性(TN), 偽陽性(FP), 偽陰性(FN)のカウント
    tp = ((y_pred == 1) & (y_test == 1)).sum().item()
    tn = ((y_pred == 0) & (y_test == 0)).sum().item()
    fp = ((y_pred == 1) & (y_test == 0)).sum().item()
    fn = ((y_pred == 0) & (y_test == 1)).sum().item()

    # 実際の「Clean」のうち、間違えてブロックした割合 (FP Rate)
    # 低いほど「誤検知が少なく、安全なサイトを壊さない」
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # 実際の「Ad/Tracker」のうち、見逃してしまった割合 (FN Rate)
    # 低いほど「防御力が高い」
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n[Security Metrics]")
    print(f"False Positive Rate (Over-blocking): {fpr:.2%}")
    print(f"False Negative Rate (Missed ads):    {fnr:.2%}")
    print(f"Overall Accuracy:                  {(tp + tn) / len(y_test):.2%}")

torch.save(classifier.state_dict(), "tracker_classifier_v1.pth")
print("Model weights saved to tracker_classifier_v1.pth")