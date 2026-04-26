"""
阶段 2 微调：从阶段 1 checkpoint 继续训练，强化问句结构
"""
import os
import time
import numpy as np
import torch
from model import GPT, GPTConfig

# ============ 配置 ============
DATA_PATH_FULL = "data/train.bin"            # 阶段 1 的全文数据
DATA_PATH_Q = "data/questions.bin"           # 阶段 2 的问句数据
CKPT_LOAD = "checkpoints/ckpt.pt"            # 加载阶段 1 模型
CKPT_SAVE = "checkpoints/ckpt_finetuned.pt"  # 保存阶段 2 模型

# 微调超参数（注意：和阶段 1 不同）
batch_size = 16              # 比阶段 1 小（数据量少）
max_iters = 600              # 步数少
eval_interval = 50           # 频繁评估
eval_iters = 10
learning_rate = 3e-5         # 关键：阶段 1 的 1/10
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 50            # 短 warmup
lr_decay_iters = 600
min_lr = 3e-6

# 混合训练比例：每个 batch 里多大概率从问句数据采样
QUESTION_RATIO = 0.7         # 70% 问句，30% 全文

# ============ 设备选择 ============
if torch.backends.mps.is_available():
    device = "mps"
    print("使用 Apple Silicon MPS 加速")
elif torch.cuda.is_available():
    device = "cuda"
    print("使用 CUDA GPU")
else:
    device = "cpu"
    print("使用 CPU")

# ============ 加载数据 ============
data_full = np.fromfile(DATA_PATH_FULL, dtype=np.uint16)
data_q = np.fromfile(DATA_PATH_Q, dtype=np.uint16)
print(f"全文数据 token 数：{len(data_full)}")
print(f"问句数据 token 数：{len(data_q)}")

# ============ 加载阶段 1 模型 ============
print(f"\n加载阶段 1 checkpoint：{CKPT_LOAD}")
ckpt = torch.load(CKPT_LOAD, map_location=device, weights_only=False)
config = ckpt["config"]
print(f"原 val loss：{ckpt['val_loss']:.4f}")
print(f"原训练步数：{ckpt['iter_num']}")

model = GPT(config)
model.load_state_dict(ckpt["model"])
model.to(device)


def get_batch():
    """混合采样：QUESTION_RATIO 的概率从问句数据采，其余从全文采"""
    block_size = config.block_size
    
    x_list = []
    y_list = []
    
    for _ in range(batch_size):
        if np.random.random() < QUESTION_RATIO:
            # 从问句数据采样
            d = data_q
        else:
            # 从全文采样
            d = data_full
        
        # 边界检查
        if len(d) <= block_size + 1:
            # 数据太短，从头开始
            i = 0
        else:
            i = np.random.randint(0, len(d) - block_size - 1)
        
        x = torch.from_numpy(d[i:i+block_size].astype(np.int64))
        y = torch.from_numpy(d[i+1:i+1+block_size].astype(np.int64))
        x_list.append(x)
        y_list.append(y)
    
    X = torch.stack(x_list).to(device)
    Y = torch.stack(y_list).to(device)
    return X, Y


def get_eval_batch(source):
    """评估专用 batch，可以指定来源"""
    block_size = config.block_size
    d = data_q if source == "questions" else data_full
    
    x_list = []
    y_list = []
    for _ in range(batch_size):
        if len(d) <= block_size + 1:
            i = 0
        else:
            i = np.random.randint(0, len(d) - block_size - 1)
        x = torch.from_numpy(d[i:i+block_size].astype(np.int64))
        y = torch.from_numpy(d[i+1:i+1+block_size].astype(np.int64))
        x_list.append(x)
        y_list.append(y)
    
    X = torch.stack(x_list).to(device)
    Y = torch.stack(y_list).to(device)
    return X, Y


@torch.no_grad()
def estimate_loss():
    """同时评估问句 loss 和全文 loss"""
    out = {}
    model.eval()
    for source in ["questions", "full"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_eval_batch(source)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[source] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ============ 优化器 ============
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
)

# ============ 训练循环 ============
print(f"\n开始微调（{max_iters} 步）...")
print(f"混合比例：{QUESTION_RATIO*100:.0f}% 问句 + {(1-QUESTION_RATIO)*100:.0f}% 全文")
print(f"初始 lr：{learning_rate}")
print()

t0 = time.time()

for iter_num in range(max_iters + 1):
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        elapsed = time.time() - t0
        print(f"step {iter_num:4d} | Q loss {losses['questions']:.4f} | full loss {losses['full']:.4f} | lr {lr:.2e} | elapsed {elapsed:.1f}s")
    
    if iter_num == max_iters:
        break
    
    X, Y = get_batch()
    _, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

# 保存最终模型
final_losses = estimate_loss()
torch.save({
    "model": model.state_dict(),
    "config": config,
    "iter_num": max_iters,
    "val_loss": final_losses["full"],
    "q_loss": final_losses["questions"],
    "stage": "finetuned",
}, CKPT_SAVE)

print(f"\n微调完成。")
print(f"最终问句 loss：{final_losses['questions']:.4f}")
print(f"最终全文 loss：{final_losses['full']:.4f}")
print(f"模型保存在：{CKPT_SAVE}")