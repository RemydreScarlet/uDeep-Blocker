import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from baselib import * # device 等の定義を想定

# Generator: 100次元のノイズから1024次元の「広告っぽい」ベクトルを生成
class Generator(nn.Module):
    def __init__(self, z_dim=100, out_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, out_dim),
            # Embeddingの範囲に合わせて調整（通常Qwen3は正規化されているためTanhが適す）
            nn.Tanh() 
        )
    def forward(self, z): return self.net(z)

# Critic (Classifier): Sigmoidを外し、スコア（スカラー）を出力するように変更
class TrackerCritic(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1) # Sigmoidなし！
        )
    def forward(self, x): return self.net(x)

# --- 補助関数: Gradient Penalty ---
def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# --- データ準備 ---
df = pd.read_pickle("js_vector_dataset.pkl")
X = np.stack(df['vector'].values)[:, :1024]
y = df['label'].values
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

# Ad(1)データのみをGeneratorの模倣対象にする
X_ad_train = torch.FloatTensor(X_train_raw[y_train_raw == 1]).to(device)
X_train = torch.FloatTensor(X_train_raw).to(device)
y_train = torch.FloatTensor(y_train_raw).view(-1, 1).to(device)
X_test = torch.FloatTensor(X_test_raw).to(device)
y_test = torch.FloatTensor(y_test_raw).view(-1, 1).to(device)

# --- 初期化 ---
z_dim = 100
gen = Generator(z_dim).to(device)
critic = TrackerCritic().to(device)
opt_gen = optim.Adam(gen.parameters(), lr=0.00001, betas=(0.5, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=0.00001, betas=(0.5, 0.9))

# --- 学習ループ ---
epochs = 500
batch_size = 64
lambda_gp = 10 # GPの係数

print("Starting WGAN-GP Training for Boundary Refinement...")
for epoch in range(epochs):
    for _ in range(5):
        # 1. データ準備
        # Real Ad / Real Clean
        idx_ad = torch.randint(0, X_ad_train.size(0), (batch_size,))
        real_ad = X_ad_train[idx_ad]
        
        idx_all = torch.randint(0, X_train.size(0), (batch_size,))
        real_batch_x = X_train[idx_all]
        real_batch_y = y_train[idx_all]

        # Fake Ad生成
        z = torch.randn(batch_size, z_dim).to(device)
        fake_ad = gen(z)

        # 2. Criticの更新 (WGAN距離 + 分類精度の両立)
        # WGAN Loss (本物Ad vs 偽物Ad)
        gp = compute_gradient_penalty(critic, real_ad, fake_ad)
        w_loss = -torch.mean(critic(real_ad)) + torch.mean(critic(fake_ad)) + lambda_gp * gp
        
        # Classification Loss (本物Ad vs 本物Clean)
        # Criticの出力をSigmoidに通してBCELossを計算
        cls_loss = nn.BCEWithLogitsLoss()(critic(real_batch_x), real_batch_y)
        
        # 合算してバックプロパゲーション
        total_d_loss = w_loss + cls_loss * 5.0 # 分類を優先的に重くする
        
        opt_critic.zero_grad()
        total_d_loss.backward()
        opt_critic.step()

    # Generatorの更新 (CriticをAdだと思い込ませる)
    z = torch.randn(batch_size, z_dim).to(device)
    fake_ad = gen(z)
    g_loss = -torch.mean(critic(fake_ad))
    
    opt_gen.zero_grad()
    g_loss.backward()
    opt_gen.step()

# --- 評価 (Criticを分類器として使用) ---
# WGANのCriticは相対的なスコアを出すため、最後に全データで境界を再調整(微調整)
print("\nFine-tuning Critic as Classifier...")
classifier_head = nn.Sequential(critic, nn.Sigmoid()).to(device)
# 最後に少しだけ分類タスクとして回す
opt_final = optim.Adam(classifier_head.parameters(), lr=0.00001)
for _ in range(100):
    opt_final.zero_grad()
    loss = nn.BCELoss()(classifier_head(X_train), y_train)
    loss.backward()
    opt_final.step()

# --- Security Metrics ---
classifier_head.eval()
with torch.no_grad():
    y_prob = classifier_head(X_test).cpu()
    y_pred = (y_prob > 0.5).float()
    
    tp = ((y_pred == 1) & (y_test.cpu() == 1)).sum().item()
    tn = ((y_pred == 0) & (y_test.cpu() == 0)).sum().item()
    fp = ((y_pred == 1) & (y_test.cpu() == 0)).sum().item()
    fn = ((y_pred == 0) & (y_test.cpu() == 1)).sum().item()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f"\n[WGAN-GP Boosted Metrics]")
    print(f"False Positive Rate: {fpr:.2%}")
    print(f"False Negative Rate: {fnr:.2%}")
    print(f"Overall Accuracy:    {(tp + tn) / len(y_test):.2%}")

torch.save(classifier_head.state_dict(), "tracker_wgan_best.pth")