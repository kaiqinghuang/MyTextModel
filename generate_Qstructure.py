"""
基于问句模式 + 全文句长分布 + 全文起手词的生成脚本

逻辑：
1. 从 patterns.json 加载：问句模式库、句长池、起手token池
2. 每次生成：
   a. 抽一个问句模式 → 决定疑问词和位置类别
   b. 抽一个目标句长（≥15 token）
   c. 抽一个起手 token
   d. 按全文风格自由生成
   e. 到达注入区间后，等标点（最多5步）后注入疑问词
   f. <NONE> 模式跳过注入
   g. 接近目标长度时软引导问号收尾
"""
import os
import random
import json
import torch
import torch.nn.functional as F
import sentencepiece as spm
from model import GPT, GPTConfig

# ============ 配置 ============
CKPT_PATH = "checkpoints/ckpt.pt"
TOKENIZER_PATH = "data/tokenizer.model"
PATTERNS_PATH = "data/patterns.json"

NUM_SAMPLES = 10
TEMPERATURE = 0.9
TOP_K = 40

MIN_LENGTH = 12           # 句长下限（token）
MAX_LENGTH = 80          # 句长上限（token）
PUNCT_WAIT_STEPS = 5      # 到达注入区间后，最多等几步找标点
QUESTION_BOOST_RANGE = 8  # 距离目标长度多少 token 时开始 boost 问号
QUESTION_BOOST_STRENGTH = 1.5  # 问号 boost 强度

# 注入位置区间（占目标长度的比例）
INJECT_RANGES = {
    "front":  (0.00, 0.30),
    "middle": (0.30, 0.70),
    "back":   (0.85, 1.00),
}

# ============ 设备 ============
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

# 收集问号 token id
stop_ids = set()
for i in range(sp.vocab_size()):
    piece = sp.id_to_piece(i)
    clean = piece.replace("▁", "")
    if clean in ["？", "?"]:
        stop_ids.add(i)
    elif (clean.endswith("？") or clean.endswith("?")) and len(clean) <= 4:
        stop_ids.add(i)

# 收集"标点 token"（中文断点，可以用作注入触发）
break_chars = set("，；：、,;:")    # 注意：不包括句号！
punct_ids = set()
for i in range(sp.vocab_size()):
    piece = sp.id_to_piece(i)
    clean = piece.replace("▁", "")
    if clean and clean[-1] in break_chars:
        punct_ids.add(i)

# 收集"陈述句终止符"（要禁止生成的，避免句子提前结束又拼接）
sentence_end_chars = set("。！.!")
sentence_end_ids = set()
for i in range(sp.vocab_size()):
    piece = sp.id_to_piece(i)
    clean = piece.replace("▁", "")
    if clean and any(c in sentence_end_chars for c in clean):
        sentence_end_ids.add(i)

# 禁止 token 集合：<Q> 标记 + 陈述句终止符
q_token_id = sp.piece_to_id("<Q>")
forbid_always_ids = set()
forbid_always_ids.add(q_token_id)
forbid_always_ids.update(sentence_end_ids)

print(f"问号 token 数：{len(stop_ids)}")
print(f"断点 token 数：{len(punct_ids)}")
print(f"始终禁止 token 数：{len(forbid_always_ids)}（含<Q>和句号等）")

print(f"问号 token 数：{len(stop_ids)}")
print(f"断点 token 数：{len(punct_ids)}")

# ============ 加载模型 ============
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
config = ckpt["config"]
model = GPT(config)
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()
print(f"模型已加载，val loss {ckpt['val_loss']:.4f}")

# ============ 加载模式库 ============
with open(PATTERNS_PATH, "r", encoding="utf-8") as f:
    patterns_data = json.load(f)

question_patterns = patterns_data["question_patterns"]
sentence_lengths = [l for l in patterns_data["sentence_lengths"] if l >= MIN_LENGTH and l <= MAX_LENGTH]
start_token_ids = patterns_data["start_token_ids"]

print(f"问句模式数：{len(question_patterns)}")
print(f"可用句长池：{len(sentence_lengths)} 条（≥{MIN_LENGTH} 且 ≤{MAX_LENGTH}）")
print(f"起手 token 池：{len(start_token_ids)}")


# ============ 采样函数 ============
def sample_one_token(model, idx, temperature, top_k,
                     forbid_ids=None, boost_ids=None, boost_value=0.0):
    """采样下一个 token，可选禁止/加权某些 token"""
    idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
    logits, _ = model(idx_cond)
    logits = logits[:, -1, :] / temperature
    
    if forbid_ids:
        for tid in forbid_ids:
            logits[:, tid] = -float('Inf')
    
    if boost_ids and boost_value > 0:
        for tid in boost_ids:
            logits[:, tid] += boost_value
    
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    return idx_next


def generate_one(model, sp, pattern, target_length,
                 temperature, top_k, stop_ids, punct_ids,
                 start_token_ids, forbid_always_ids):
    """
    按一个模式生成一个句子
    """
    qw = pattern["question_word"]
    position = pattern["position"]
    
    # 计算注入区间
    if qw == "<NONE>" or position == "none":
        inject_needed = False
        inject_start = -1
        inject_end = -1
        qw_ids = []
    else:
        inject_needed = True
        ratio_lo, ratio_hi = INJECT_RANGES[position]
        inject_start = int(target_length * ratio_lo)
        inject_end = int(target_length * ratio_hi)
        qw_ids = sp.encode(qw)
    
    # 起手 token
    start_id = random.choice(start_token_ids)
    idx = torch.tensor([[start_id]], dtype=torch.long, device=device)
    
    qw_injected = (not inject_needed)
    punct_wait_count = 0
    
    # 用于 back 模式：注入后强制问号
    force_stop_countdown = -1
    
    while True:
        current_pos = idx.size(1)
        
        if current_pos >= MAX_LENGTH:
            break
        
        # back 注入后强制问号收尾
        if force_stop_countdown >= 0:
            # back 注入后，给 stop 强 boost，倒计时归零时直接挑一个 stop token
            if force_stop_countdown == 0:
                # 直接挑一个最常见的问号 token
                stop_id = next(iter(stop_ids))
                idx = torch.cat((idx, torch.tensor([[stop_id]], dtype=torch.long, device=device)), dim=1)
                break
            
            idx_next = sample_one_token(
                model, idx, temperature, top_k,
                forbid_ids=forbid_always_ids,
                boost_ids=stop_ids, boost_value=5.0,
            )
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() in stop_ids:
                break
            force_stop_countdown -= 1
            continue
        
        # 正常生成阶段
        forbid_now = set(forbid_always_ids)
        
        if inject_needed and not qw_injected:
            if current_pos < inject_start:
                # 还没到注入区间：自由生成 + 禁止停止
                forbid_now.update(stop_ids)
                idx_next = sample_one_token(
                    model, idx, temperature, top_k,
                    forbid_ids=forbid_now,
                )
                idx = torch.cat((idx, idx_next), dim=1)
            
            elif current_pos < inject_end:
                # 在注入区间：等标点
                forbid_now.update(stop_ids)
                idx_next = sample_one_token(
                    model, idx, temperature, top_k,
                    forbid_ids=forbid_now,
                )
                idx = torch.cat((idx, idx_next), dim=1)
                
                if idx_next.item() in punct_ids:
                    qw_tensor = torch.tensor([qw_ids], dtype=torch.long, device=device)
                    idx = torch.cat((idx, qw_tensor), dim=1)
                    qw_injected = True
                    if position == "back":
                        force_stop_countdown = 3
                else:
                    punct_wait_count += 1
                    if punct_wait_count >= PUNCT_WAIT_STEPS:
                        qw_tensor = torch.tensor([qw_ids], dtype=torch.long, device=device)
                        idx = torch.cat((idx, qw_tensor), dim=1)
                        qw_injected = True
                        if position == "back":
                            force_stop_countdown = 3
            
            else:
                # 超过区间还没注入，强制注入
                qw_tensor = torch.tensor([qw_ids], dtype=torch.long, device=device)
                idx = torch.cat((idx, qw_tensor), dim=1)
                qw_injected = True
                if position == "back":
                    force_stop_countdown = 3
        
        else:
            # 已注入或 NONE 模式，继续生成到目标长度
            tokens_to_target = target_length - current_pos
            
            if tokens_to_target <= QUESTION_BOOST_RANGE:
                boost = QUESTION_BOOST_STRENGTH * (QUESTION_BOOST_RANGE - tokens_to_target + 1) / QUESTION_BOOST_RANGE
                idx_next = sample_one_token(
                    model, idx, temperature, top_k,
                    forbid_ids=forbid_now,
                    boost_ids=stop_ids, boost_value=boost,
                )
            else:
                forbid_now.update(stop_ids)
                idx_next = sample_one_token(
                    model, idx, temperature, top_k,
                    forbid_ids=forbid_now,
                )
            
            idx = torch.cat((idx, idx_next), dim=1)
            
            if idx_next.item() in stop_ids:
                break
    
    return idx


# ============ 主循环 ============
print(f"\n开始生成 {NUM_SAMPLES} 个问句\n" + "=" * 60)

for i in range(NUM_SAMPLES):
    pattern = random.choice(question_patterns)
    target_length = random.choice(sentence_lengths)
    
    out = generate_one(
        model, sp, pattern, target_length,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        stop_ids=stop_ids,
        punct_ids=punct_ids,
        start_token_ids=start_token_ids,
        forbid_always_ids=forbid_always_ids,
    )
    
    text = sp.decode(out[0].tolist()).strip()
    if not (text.endswith("？") or text.endswith("?")):
        text += "？"
    
    qw_display = pattern["question_word"]
    pos_display = pattern["position"]
    print(f"[{i+1:2d}] 模式：{qw_display}@{pos_display} | 目标长度：{target_length}")
    print(f"     生成：{text}\n")

print("=" * 60)