"""
推理脚本：从<Q>开始生成问句
"""
import os
import sys
import torch
import sentencepiece as spm
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parent / "train"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))
from _paths import DATA_DIR, CHECKPOINTS_DIR
from model import GPT, GPTConfig

# ============ 配置 ============
CKPT_PATH = str(CHECKPOINTS_DIR / "ckpt.pt")
TOKENIZER_PATH = str(DATA_DIR / "tokenizer.model")

# 生成参数
num_samples = 10           # 生成多少个问句
max_new_tokens = 120        # 单个问句最大长度
temperature = 1.2         # 温度：0.5冷静 / 1.0标准 / 1.2狂野
top_k = 40                 # 只从概率top-k个token里采样

# ============ 设备选择 ============
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"使用设备：{device}")

# ============ 加载分词器 ============
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

q_token_id = sp.piece_to_id("<Q>")
print(f"<Q> token id: {q_token_id}")

# 终止 token：中英文问号
stop_ids = set()
for ch in ["？", "?"]:
    ids = sp.encode(ch)
    for i in ids:
        stop_ids.add(i)
# 也找出包含问号的所有token（BPE可能把问号合并到其他piece里）
for i in range(sp.vocab_size()):
    piece = sp.id_to_piece(i)
    if "？" in piece or "?" in piece:
        stop_ids.add(i)
print(f"终止 token 数量：{len(stop_ids)}")

# ============ 加载模型 ============
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
config = ckpt["config"]
model = GPT(config)
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()
print(f"模型已加载，来自 step {ckpt['iter_num']}，val loss {ckpt['val_loss']:.4f}")

# ============ 生成 ============
print(f"\n开始生成 {num_samples} 个问句\n" + "="*50)

for i in range(num_samples):
    # 起始 token：<Q>
    idx = torch.tensor([[q_token_id]], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_ids=stop_ids,
    )
    # 解码
    token_ids = out[0].tolist()
    # 去掉开头的 <Q>，只解码后面的内容
    text = sp.decode(token_ids[1:])
    print(f"[{i+1:2d}] {text}")

print("="*50 + "\n生成完成")