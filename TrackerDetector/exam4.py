import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from baselib import * # device 等の定義を想定

# --- GPU初期化（cuBLAS警告対策） ---
if torch.cuda.is_available():
    torch.cuda.init()

# --- Generator ---
# Ad用とClean用の2つのインスタンスを作成するため、クラスは1つでOK
class Generator(nn.Module):
    def __init__(self, z_dim=100, out_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, out_dim),
            nn.Tanh() # Qwen3の正規化ベクトルに合わせる
        )
    def forward(self, z): return self.net(z)

# --- Critic (判定器) ---
# スコアが大きいほどAd、小さいほどCleanと判定する
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
            nn.Linear(128, 1) # Sigmoidなし（WGANスコア用）
        )
    def forward(self, x): return self.net(x)

# --- 勾配ペナルティ (GP) の計算 ---
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
print("Loading dataset...")
df = pd.read_pickle("js_vector_dataset.pkl")
X = np.stack(df['vector'].values)[:, :1024]
y = df['label'].values
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train_raw).to(device)
y_train = torch.FloatTensor(y_train_raw).view(-1, 1).to(device)
X_test = torch.FloatTensor(X_test_raw).to(device)
y_test = torch.FloatTensor(y_test_raw).view(-1, 1).to(device)

# Ad(1) と Clean(0) を分離
X_ad_train = torch.FloatTensor(X_train_raw[y_train_raw == 1]).to(device)
X_clean_train = torch.FloatTensor(X_train_raw[y_train_raw == 0]).to(device)

# --- 初期化 ---
z_dim = 100
gen_ad = Generator(z_dim).to(device)     # 広告の境界を広げるGenerator
gen_clean = Generator(z_dim).to(device)  # 正常の境界を広げるGenerator
critic = TrackerCritic().to(device)

opt_gen_ad = optim.Adam(gen_ad.parameters(), lr=0.00001, betas=(0.5, 0.9))
opt_gen_clean = optim.Adam(gen_clean.parameters(), lr=0.00001, betas=(0.5, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=0.00001, betas=(0.5, 0.9))

# --- 学習ループ ---
epochs = 500
batch_size = 64
lambda_gp = 10

print("Starting Dual-GAN Training for Surgical Precision...")
for epoch in range(epochs):
    for _ in range(5):
        # 1. データの準備
        idx_ad = torch.randint(0, X_ad_train.size(0), (batch_size,))
        real_ad = X_ad_train[idx_ad]
        
        idx_clean = torch.randint(0, X_clean_train.size(0), (batch_size,))
        real_clean = X_clean_train[idx_clean]
        
        idx_all = torch.randint(0, X_train.size(0), (batch_size,))
        real_batch_x = X_train[idx_all]
        real_batch_y = y_train[idx_all]

        # Fakeの生成
        z_ad = torch.randn(batch_size, z_dim).to(device)
        fake_ad = gen_ad(z_ad)
        
        z_clean = torch.randn(batch_size, z_dim).to(device)
        fake_clean = gen_clean(z_clean)

        # 2. Criticの更新
        gp_ad = compute_gradient_penalty(critic, real_ad, fake_ad)
        w_loss_ad = -torch.mean(critic(real_ad)) + torch.mean(critic(fake_ad)) + lambda_gp * gp_ad
        
        gp_clean = compute_gradient_penalty(critic, real_clean, fake_clean)
        w_loss_clean = torch.mean(critic(real_clean)) - torch.mean(critic(fake_clean)) + lambda_gp * gp_clean
        
        logits = critic(real_batch_x)
        bce_loss_unreduced = nn.BCEWithLogitsLoss(reduction='none')(logits, real_batch_y)
        
        weights = torch.where(real_batch_y == 0, 5.0, 1.0)
        cls_loss = (bce_loss_unreduced * weights).mean()
        
        total_d_loss = w_loss_ad + w_loss_clean + cls_loss * 5.0
        
        opt_critic.zero_grad()
        total_d_loss.backward()
        opt_critic.step()

    # 3. Generatorsの更新
    z_ad = torch.randn(batch_size, z_dim).to(device)
    fake_ad = gen_ad(z_ad)
    g_loss_ad = -torch.mean(critic(fake_ad))
    
    z_clean = torch.randn(batch_size, z_dim).to(device)
    fake_clean = gen_clean(z_clean)
    g_loss_clean = torch.mean(critic(fake_clean)) # Fake(Clean)のスコアを小さくしたい
    
    opt_gen_ad.zero_grad()
    g_loss_ad.backward()
    opt_gen_ad.step()
    
    opt_gen_clean.zero_grad()
    g_loss_clean.backward()
    opt_gen_clean.step()

# --- 最終微調整 (Fine-tuning) ---
print("\nFine-tuning Critic as Classifier with High FP-Penalty...")
classifier_head = nn.Sequential(critic, nn.Sigmoid()).to(device)
opt_final = optim.Adam(classifier_head.parameters(), lr=0.00001)

# ここでもClean(正常)を重視して最終学習
for _ in range(100):
    opt_final.zero_grad()
    probs = classifier_head(X_train)
    bce = nn.BCELoss(reduction='none')(probs, y_train)
    weights_ft = torch.where(y_train == 0, 5.0, 1.0) # FPペナルティ5倍
    loss = (bce * weights_ft).mean()
    loss.backward()
    opt_final.step()

# --- セキュリティメトリクスの評価 ---
classifier_head.eval()
with torch.no_grad():
    y_prob = classifier_head(X_test).cpu()
    
    threshold = 0.5
    y_pred = (y_prob > threshold).float()
    
    tp = ((y_pred == 1) & (y_test.cpu() == 1)).sum().item()
    tn = ((y_pred == 0) & (y_test.cpu() == 0)).sum().item()
    fp = ((y_pred == 1) & (y_test.cpu() == 0)).sum().item()
    fn = ((y_pred == 0) & (y_test.cpu() == 1)).sum().item()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    accuracy = (tp + tn) / len(y_test)
    
    print(f"\n[Dual-GAN Surgical Metrics (Threshold: {threshold})]")
    print(f"False Positive Rate: {fpr:.2%}")
    print(f"False Negative Rate: {fnr:.2%}")
    print(f"Overall Accuracy:    {accuracy:.2%}")

torch.save(classifier_head.state_dict(), "tracker_dual_gan_best.pth")
print("\nModel saved as 'tracker_dual_gan_best.pth'")