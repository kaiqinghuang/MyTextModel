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
PUNCT_WAIT_STEPS = 8      # 到达注入区间后，最多等几步找标点
QUESTION_BOOST_RANGE = 8  # 距离目标长度多少 token 时开始 boost 问号
QUESTION_BOOST_STRENGTH = 1.5  # 问号 boost 强度

# 注入位置区间（占目标长度的比例）
# front 和 back 的区间不再使用，逻辑由 generate_one 单独处理
INJECT_RANGES = {
    "middle": (0.20, 0.80),
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
                     forbid_ids=None, boost_ids=None, boost_value=0.0,
                     repetition_penalty=1.3, repetition_window=20,
                     hard_block_consecutive=3):
    """
    采样下一个 token
    
    repetition_penalty: 对 window 内出现过的 token 降权（>1 表示降权）
    repetition_window: 重复检测窗口
    hard_block_consecutive: 一个 token 连续出现达到这个次数时，硬禁止它再出现
    """
    idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
    logits, _ = model(idx_cond)
    logits = logits[:, -1, :] / temperature
    
    # 软惩罚：window 内出现过的 token 降权
    if repetition_penalty > 1.0 and idx.size(1) > 0:
        recent_window = idx[0, -repetition_window:].tolist()
        recent_set = set(recent_window)
        for tid in recent_set:
            if logits[0, tid] > 0:
                logits[0, tid] /= repetition_penalty
            else:
                logits[0, tid] *= repetition_penalty
    
    # 硬截断：连续出现 N 次的 token 完全禁止
    if hard_block_consecutive > 0 and idx.size(1) >= hard_block_consecutive:
        # 检查最后 N 个 token 是否相同
        last_n = idx[0, -hard_block_consecutive:].tolist()
        if len(set(last_n)) == 1:
            # 全部相同，禁止它继续出现
            blocked_id = last_n[0]
            logits[:, blocked_id] = -float('Inf')
    
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

def maybe_inject_qw(idx, qw, qw_ids, sp, check_window=8):
    """
    检查最后几个 token 是否已经包含疑问词，没有的话注入
    返回：(更新后的 idx, 是否注入了)
    """
    actual_window = min(check_window, idx.size(1))
    recent_text = sp.decode(idx[0, -actual_window:].tolist())
    already_has_qw = qw in recent_text
    
    if not already_has_qw:
        qw_tensor = torch.tensor([qw_ids], dtype=torch.long, device=device)
        idx = torch.cat((idx, qw_tensor), dim=1)
    
    return idx, not already_has_qw

def generate_one(model, sp, pattern, target_length,
                 temperature, top_k, stop_ids, punct_ids,
                 start_token_ids, forbid_always_ids):
    """
    按一个模式生成一个句子
    front: 疑问词作为起手词
    back: 疑问词作为末尾词，紧接问号
    middle: 在 20%-80% 区间内注入，优先标点后
    none: 不注入，自由生成 + 软引导问号
    """
    qw = pattern["question_word"]
    position = pattern["position"]
    
    # ============ 模式分支 1：<NONE> ============
    if qw == "<NONE>" or position == "none":
        # 起手 token 从池子里随机抽
        start_id = random.choice(start_token_ids)
        idx = torch.tensor([[start_id]], dtype=torch.long, device=device)
        
        while True:
            current_pos = idx.size(1)
            if current_pos >= MAX_LENGTH:
                break
            
            forbid_now = set(forbid_always_ids)
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
    
    # ============ 模式分支 2：front - 疑问词作为起手 ============
    if position == "front":
        qw_ids = sp.encode(qw)
        idx = torch.tensor([qw_ids], dtype=torch.long, device=device)
        
        # 自由生成到目标长度，软引导问号收尾
        while True:
            current_pos = idx.size(1)
            if current_pos >= MAX_LENGTH:
                break
            
            forbid_now = set(forbid_always_ids)
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
    
    # ============ 模式分支 3：back - 疑问词作为末尾 ============
    if position == "back":
        qw_ids = sp.encode(qw)
        # 起手 token 从池子里随机抽
        start_id = random.choice(start_token_ids)
        idx = torch.tensor([[start_id]], dtype=torch.long, device=device)
        
        # 自由生成到 (target_length - len(qw_ids) - 1) 时停下来
        target_before_qw = max(2, target_length - len(qw_ids) - 1)
        
        while idx.size(1) < target_before_qw:
            if idx.size(1) >= MAX_LENGTH:
                break
            
            forbid_now = set(forbid_always_ids)
            forbid_now.update(stop_ids)  # 这阶段禁止停止 token
            
            idx_next = sample_one_token(
                model, idx, temperature, top_k,
                forbid_ids=forbid_now,
            )
            idx = torch.cat((idx, idx_next), dim=1)
        
        # 注入疑问词（如果最近没出现过）
        idx, _ = maybe_inject_qw(idx, qw, qw_ids, sp)
        
        # 紧接问号
        forbid_now = set(forbid_always_ids)
        idx_next = sample_one_token(
            model, idx, temperature, top_k,
            forbid_ids=forbid_now,
            boost_ids=stop_ids, boost_value=10.0,
        )
        idx = torch.cat((idx, idx_next), dim=1)
        # 兜底
        if idx_next.item() not in stop_ids:
            stop_id = next(iter(stop_ids))
            idx = torch.cat((idx, torch.tensor([[stop_id]], dtype=torch.long, device=device)), dim=1)
        
        return idx
    
    # ============ 模式分支 4：middle - 区间内注入 ============
    if position == "middle":
        qw_ids = sp.encode(qw)
        # 起手 token 从池子里随机抽
        start_id = random.choice(start_token_ids)
        idx = torch.tensor([[start_id]], dtype=torch.long, device=device)
        
        ratio_lo, ratio_hi = INJECT_RANGES["middle"]
        inject_start = int(target_length * ratio_lo)
        inject_end = int(target_length * ratio_hi)
        
        qw_injected = False
        punct_wait_count = 0
        
        while True:
            current_pos = idx.size(1)
            if current_pos >= MAX_LENGTH:
                break
            
            if not qw_injected:
                if current_pos < inject_start:
                    # 还没到注入区间
                    forbid_now = set(forbid_always_ids)
                    forbid_now.update(stop_ids)
                    idx_next = sample_one_token(
                        model, idx, temperature, top_k,
                        forbid_ids=forbid_now,
                    )
                    idx = torch.cat((idx, idx_next), dim=1)
                
                elif current_pos < inject_end:
                    # 在注入区间，等标点
                    forbid_now = set(forbid_always_ids)
                    forbid_now.update(stop_ids)
                    idx_next = sample_one_token(
                        model, idx, temperature, top_k,
                        forbid_ids=forbid_now,
                    )
                    idx = torch.cat((idx, idx_next), dim=1)
                    
                    if idx_next.item() in punct_ids:
                        idx, _ = maybe_inject_qw(idx, qw, qw_ids, sp)
                        qw_injected = True
                        
                    else:
                        punct_wait_count += 1
                        if punct_wait_count >= PUNCT_WAIT_STEPS:
                            idx, _ = maybe_inject_qw(idx, qw, qw_ids, sp)
                            qw_injected = True
                            
                
                else:
                    idx, _ = maybe_inject_qw(idx, qw, qw_ids, sp)
                    qw_injected = True
                    
            
            else:
                # 已注入，继续生成到目标长度，软引导问号
                forbid_now = set(forbid_always_ids)
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
    
    # 兜底：不应该到这里
    raise ValueError(f"未知的 position: {position}")


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