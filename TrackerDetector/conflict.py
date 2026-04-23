import os
import hashlib
import pandas as pd

# 設定：JSファイルが格納されているディレクトリ
AD_DIR = "./dataset/ad_tracker"
CLEAN_DIR = "./dataset/clean"

def get_file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def clean_dataset():
    ad_hashes = {}    # hash -> filename
    clean_hashes = {} # hash -> filename
    
    # 1. Adファイルのハッシュ収集
    for f in os.listdir(AD_DIR):
        if f.endswith('.js'):
            h = get_file_hash(os.path.join(AD_DIR, f))
            ad_hashes[h] = f

    # 2. Cleanファイルのハッシュ収集
    for f in os.listdir(CLEAN_DIR):
        if f.endswith('.js'):
            h = get_file_hash(os.path.join(CLEAN_DIR, f))
            clean_hashes[h] = f

    # 3. 衝突（衝突＝同じファイルが両方にある）の特定
    conflicts = set(ad_hashes.keys()) & set(clean_hashes.keys())
    
    print(f"--- Dataset Statistics ---")
    print(f"Total Ad files:    {len(ad_hashes)}")
    print(f"Total Clean files: {len(clean_hashes)}")
    print(f"Conflicting files: {len(conflicts)} (Found in BOTH folders)")

    if conflicts:
        print("\n[Action] Removing conflicts from the list...")
        for h in conflicts:
            print(f"Conflict found: {ad_hashes[h]} <--> {clean_hashes[h]}")
            # Ads側から削除する
            del ad_hashes[h]
    
    # 4. 重複を除いた「真の学習リスト」を作成
    final_list = []
    for h, f in ad_hashes.items():
        final_list.append({'file': f, 'path': os.path.join(AD_DIR, f), 'label': 1})
            
    for h, f in clean_hashes.items():
        final_list.append({'file': f, 'path': os.path.join(CLEAN_DIR, f), 'label': 0})

    return pd.DataFrame(final_list)

# 実行
df_clean = clean_dataset()
df_clean.to_csv("verified_js_list.csv", index=False)
print(f"\nSaved {len(df_clean)} verified samples to verified_js_list.csv")