import os
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# VRAMの断片化を防ぐ設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class JSBatchEmbedder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", batch_size=1):
        # RTX 2070 Super (cuda:1) を優先使用
        self.device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        self.batch_size = batch_size
        
        print(f"Loading model on {self.device} with 4-bit quantization...")
        
        # 4-bit量子化の設定
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map={"": self.device} # モデル全体を指定したGPUに配置
        )
        self.model.eval()

    def get_embeddings_batch(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, 
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Maskの準備
            attention_mask = inputs['attention_mask']
            last_hidden = outputs.last_hidden_state # [batch, seq_len, 1024]
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()

            # Mean Pooling (全体の統計)
            sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_pool = sum_embeddings / sum_mask

            # Max Pooling (決定的な特徴の抽出)
            # マスクされていない部分の最大値を取るため、無効な部分は負の大きな値で埋める
            ignored_mask = (1.0 - mask_expanded) * -1e9
            max_pool = torch.max(last_hidden + ignored_mask, 1)[0]
            
            # Concat (1024 + 1024 = 2048次元)
            result = torch.cat([mean_pool, max_pool], dim=1).cpu()
            
        torch.cuda.empty_cache()
        return result

    def process_directory(self, root_dir="dataset"):
        all_data = []
        for label_name, label_val in [("ad_tracker", 1), ("clean", 0)]:
            dir_path = os.path.join(root_dir, label_name)
            if not os.path.exists(dir_path): continue
            
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".js")]
            
            for i in tqdm(range(0, len(files), self.batch_size), desc=f"Processing {label_name}"):
                batch_files = files[i : i + self.batch_size]
                batch_texts = []
                for f_path in batch_files:
                    try:
                        with open(f_path, "r", encoding="utf-8", errors="ignore") as f:
                            batch_texts.append(f.read())
                    except:
                        continue
                
                if not batch_texts: continue
                
                try:
                    embeddings = self.get_embeddings_batch(batch_texts)
                    for j, vec in enumerate(embeddings):
                        all_data.append({
                            "vector": vec.numpy(),
                            "label": label_val,
                            "file": os.path.basename(batch_files[j])
                        })
                except torch.cuda.OutOfMemoryError:
                    print(f"\nSkipping batch due to OOM: {batch_files}")
                    torch.cuda.empty_cache()
                    continue

        df = pd.DataFrame(all_data)
        df.to_pickle("js_vector_dataset.pkl")
        print(f"Dataset saved! Total samples: {len(df)}")

if __name__ == "__main__":
    embedder = JSBatchEmbedder(batch_size=1) 
    embedder.process_directory()