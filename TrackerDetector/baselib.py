import torch
from transformers import AutoModel, AutoTokenizer

model_name = "Qwen/Qwen3-Embedding-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

# Mean PoolingとMax Pooling
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        
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
