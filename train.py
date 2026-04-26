"""
训练脚本
"""
import os
import time
import numpy as np
import torch
from model import GPT, GPTConfig

# ============ 配置 ============
DATA_PATH = "data/train.bin"
CKPT_DIR = "checkpoints"
CKPT_PATH = os.path.join(CKPT_DIR, "ckpt.pt")

# 模型超参数
config = GPTConfig(
    block_size=128,
    vocab_size=5000,
    n_layer=6,
    n_head=6,
    n_embd=192,
    dropout=0.15,
    bias=True,
)

# 训练超参数
batch_size = 32
max_iters = 8000
eval_interval = 200       # 每多少步打印一次loss
eval_iters = 20           # 评估时采样多少个batch求平均
learning_rate = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 200        # 学习率warmup步数
lr_decay_iters = 8000     # 学习率衰减总步数
min_lr = 3e-5

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

# ============ 数据加载 ============
data = np.fromfile(DATA_PATH, dtype=np.uint16)
print(f"数据加载完成，token总数：{len(data)}")

# 90% 训练，10% 验证
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"训练集：{len(train_data)} tokens，验证集：{len(val_data)} tokens")


def get_batch(split):
    """随机采样一个batch"""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - config.block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(d[i:i+config.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i+1:i+1+config.block_size].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """评估train和val的平均loss"""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
    """学习率调度：warmup + cosine decay"""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ============ 模型和优化器 ============
os.makedirs(CKPT_DIR, exist_ok=True)
model = GPT(config)
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
)

# ============ 训练循环 ============
print("\n开始训练...")
t0 = time.time()

for iter_num in range(max_iters + 1):
    # 调整学习率
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # 评估
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        elapsed = time.time() - t0
        print(f"step {iter_num:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e} | elapsed {elapsed:.1f}s")
        

    if iter_num == max_iters:
        break

    # 训练一步
    X, Y = get_batch("train")
    _, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

# 训练结束后保存最终模型
final_losses = estimate_loss()
torch.save({
    "model": model.state_dict(),
    "config": config,
    "iter_num": max_iters,
    "val_loss": final_losses["val"],
    "train_loss": final_losses["train"],
}, CKPT_PATH)

print(f"\n训练完成。")
print(f"最终 train loss：{final_losses['train']:.4f}")
print(f"最终 val loss：{final_losses['val']:.4f}")
print(f"模型保存在：{CKPT_PATH}")