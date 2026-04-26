"""
扫描 marked.txt：
1. 抽取所有问句的疑问词 + 位置类别 → 问句模式库
2. 抽取所有句子的长度分布 → 句长分布池
"""
import re
import json
import jieba
import sentencepiece as spm

MARKED_PATH = "data/marked.txt"
TOKENIZER_PATH = "data/tokenizer.model"
OUTPUT_PATH = "data/patterns.json"

# 疑问词词典（不追求完备，覆盖常见用法即可）
QUESTION_WORDS = {
    # 疑问代词
    "什么", "为什么", "为何", "怎么", "怎样", "如何",
    "哪里", "哪儿", "哪", "谁", "几", "多少",
    # 疑问副词
    "难道", "到底", "究竟", "莫非",
    # 是非问的固定结构
    "是不是", "有没有", "能不能", "可不可以", "会不会",
    "要不要", "好不好", "行不行", "对不对",
    "是否", "可否", "能否",
    # 句末语气词
    "吗", "呢", "吧", "啊", "呀",
}

# 加载分词器
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)


def find_question_word(text):
    """
    用 jieba 切词，然后从词列表里找疑问词。
    返回 (疑问词, 字符位置) 或 (None, -1)。
    优先返回出现位置最靠前的、且最长的疑问词。
    """
    words = list(jieba.cut(text))
    char_pos = 0
    found = []
    for w in words:
        if w in QUESTION_WORDS:
            found.append((w, char_pos))
        char_pos += len(w)
    
    if not found:
        return None, -1
    
    # 优先：靠前的；同位置取更长的
    found.sort(key=lambda x: (x[1], -len(x[0])))
    return found[0]


def categorize_position(char_pos, total_chars):
    """位置类别：前/中/后"""
    if char_pos < 0:
        return "none"
    ratio = char_pos / total_chars
    if ratio < 0.33:
        return "front"
    elif ratio < 0.67:
        return "middle"
    else:
        return "back"


# ============ 主流程 ============

# 读取 marked.txt
with open(MARKED_PATH, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

question_patterns = []      # 问句模式库
sentence_lengths = []       # 全文所有句子的 token 长度
start_token_ids = []        # 所有句子的首 token 池

for line in lines:
    line = line.strip()
    if not line:
        continue
    
    # 判断是否问句
    is_question = line.startswith("<Q>")
    
    # 去掉 <Q> 前缀（如果有），得到纯句子
    if is_question:
        pure = line[3:]
    else:
        pure = line
    
    # 句长（token 数）
    token_ids = sp.encode(pure)
    token_count = len(token_ids)
    if 3 <= token_count <= 80:
        sentence_lengths.append(token_count)
    
    # 提取首 token（跳过空白/标点 piece）
    for tid in token_ids:
        piece = sp.id_to_piece(tid)
        clean = piece.replace("▁", "").strip()
        # 跳过空 piece、纯标点
        if not clean:
            continue
        if all(not c.isalnum() and c not in "一二三四五六七八九十百千万亿" 
               and not ('\u4e00' <= c <= '\u9fff') for c in clean):
            continue
        start_token_ids.append(tid)
        break  # 只取首个有效 token
    
    # 如果是问句，提取疑问词模式
    if is_question and len(pure) >= 3:
        # 去掉末尾问号方便定位
        text_for_qw = pure.rstrip("？?")
        if not text_for_qw:
            continue
        
        qw, char_pos = find_question_word(text_for_qw)
        position = categorize_position(char_pos, len(text_for_qw))
        
        question_patterns.append({
            "question_word": qw if qw else "<NONE>",
            "position": position,
            "original": pure,
        })

# 保存
output = {
    "question_patterns": question_patterns,
    "sentence_lengths": sentence_lengths,
    "start_token_ids": start_token_ids,
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# ============ 统计输出 ============
from collections import Counter

print(f"问句模式总数：{len(question_patterns)}")
print(f"全文句子总数：{len(sentence_lengths)}")

qw_counter = Counter(p["question_word"] for p in question_patterns)
pos_counter = Counter(p["position"] for p in question_patterns)

print(f"\n疑问词 Top 15：")
for qw, count in qw_counter.most_common(15):
    print(f"  {qw}: {count}")

print(f"\n疑问词位置分布：")
for pos, count in pos_counter.most_common():
    print(f"  {pos}: {count}")

print(f"\n句长（token）分布：")
print(f"  最短：{min(sentence_lengths)}")
print(f"  最长：{max(sentence_lengths)}")
print(f"  平均：{sum(sentence_lengths)/len(sentence_lengths):.1f}")
print(f"  中位数：{sorted(sentence_lengths)[len(sentence_lengths)//2]}")

# 长度分桶
buckets = {"3-10": 0, "11-20": 0, "21-30": 0, "31-50": 0, "51-80": 0}
for l in sentence_lengths:
    if l <= 10:
        buckets["3-10"] += 1
    elif l <= 20:
        buckets["11-20"] += 1
    elif l <= 30:
        buckets["21-30"] += 1
    elif l <= 50:
        buckets["31-50"] += 1
    else:
        buckets["51-80"] += 1
for k, v in buckets.items():
    print(f"  {k} tokens: {v}")

print(f"\n输出文件：{OUTPUT_PATH}")
# 在 print(f"\n输出文件：...") 之前加这段
print(f"\n起手 token 池大小：{len(start_token_ids)}")
unique_starts = len(set(start_token_ids))
print(f"  唯一首 token 数：{unique_starts}")
top_starts = Counter(start_token_ids).most_common(10)
print(f"  Top 10 起手 token：")
for tid, count in top_starts:
    print(f"    [{tid}] '{sp.id_to_piece(tid)}': {count}")