"""
从 marked.txt 抽取所有问句，编码成 questions.bin
用于阶段 2 微调
"""
import numpy as np
import sentencepiece as spm

MARKED_PATH = "data/marked.txt"
TOKENIZER_PATH = "data/tokenizer.model"
OUTPUT_PATH = "data/questions.bin"

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

# 读取所有问句
questions = []
with open(MARKED_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("<Q>"):
            questions.append(line)

print(f"问句总数：{len(questions)}")

# 把所有问句串起来，每条之间用换行符分隔
# 这样训练时，模型会学习"<Q>...?\n<Q>...?\n" 这种序列模式
joined_text = "\n".join(questions)

# 编码
ids = sp.encode(joined_text)
print(f"编码后 token 总数：{len(ids)}")

# 检查
print(f"<Q> token id: {sp.piece_to_id('<Q>')}")
q_count_in_ids = ids.count(sp.piece_to_id("<Q>"))
print(f"编码后 <Q> 出现次数：{q_count_in_ids}（应该 ≈ {len(questions)}）")

# 保存
ids_array = np.array(ids, dtype=np.uint16)
ids_array.tofile(OUTPUT_PATH)
print(f"\n保存到：{OUTPUT_PATH}")
print(f"文件大小：{ids_array.nbytes/1024:.2f} KB")

# 抽几个样本展示
print("\n前 3 条问句：")
for q in questions[:3]:
    print(f"  {q}")