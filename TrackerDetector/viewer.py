import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. データセットの読み込み
print("Loading dataset: js_vector_dataset.pkl...")
df = pd.read_pickle("js_vector_dataset.pkl")

X = np.stack(df['vector'].values)
y = df['label'].values

# 2. t-SNEによる次元圧縮 (引数をシンプルに修正)
print(f"Running t-SNE on {len(X)} samples...")
tsne = TSNE(
    n_components=2, 
    perplexity=30, 
    random_state=42,
    init='random',      # pcaでエラーが出る場合があるのでrandomに
    learning_rate=200.0 # 数値を直接指定（あるいは 'auto' が通るなら 'auto'）
)
X_embedded = tsne.fit_transform(X)

# 3. 可視化
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# 1: 赤(Ad/Tracker), 0: 青(Clean)
colors = {1: "#e74c3c", 0: "#3498db"}
labels = {1: "Ad/Tracker JS", 0: "Clean JS"}

for label_val in [0, 1]:
    mask = (y == label_val)
    plt.scatter(
        X_embedded[mask, 0], 
        X_embedded[mask, 1], 
        c=colors[label_val], 
        label=labels[label_val],
        alpha=0.6, 
        edgecolors='none', 
        s=40
    )

plt.legend()
plt.title(f"Qwen3-Embedding: JS Intent Distribution (n={len(X)})", fontsize=15)
plt.savefig("js_distribution_tsne.png", dpi=300)
print("Saved: js_distribution_tsne.png")
plt.show()

# 4. 精度評価
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print("\n" + "="*30)
print(f"Linear Classification Accuracy: {score:.2%}")
print("="*30)
print(classification_report(y_test, clf.predict(X_test)))