# とりあえずコサイン類似度で判別してみる


from torch.nn.functional import cosine_similarity
from baselib import *

model_name = "Qwen/Qwen3-Embedding-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

# 平均（Mean Pooling）をとる
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        attention_mask = inputs['attention_mask']
        last_hidden = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


normal_js = """
function add(a, b) { return a + b; }
console.log("Hello World");
const data = [1, 2, 3].map(x => x * 2);
"""

tracker_js = """
const fingerprint = () => {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  ctx.fillText('ads_tracker', 2, 2);
  return canvas.toDataURL();
};
fetch('https://tracker.com/collect', { body: fingerprint() });
"""

unknown_js = """
(function(_0x123){
  // 難読化された風のコード
  const _0xabc = window.localStorage.getItem('user_id');
  navigator.sendBeacon('https://api.stat-collector.io/v1', _0xabc);
})()
"""

print("--- 特徴抽出開始 ---")
vec_normal = get_embedding(normal_js)
vec_tracker = get_embedding(tracker_js)
vec_unknown = get_embedding(unknown_js)

sim_to_normal = cosine_similarity(vec_unknown, vec_normal).item()
sim_to_tracker = cosine_similarity(vec_unknown, vec_tracker).item()

print(f"未知のJS vs 正常JS の類似度: {sim_to_normal:.4f}")
print(f"未知のJS vs 追跡JS の類似度: {sim_to_tracker:.4f}")

if sim_to_tracker > sim_to_normal:
    print("\n【判定結果】広告・トラッカーの可能性が高いです (Block推奨)")
else:
    print("\n【判定結果】正常なスクリプトの可能性が高いです (Allow)")