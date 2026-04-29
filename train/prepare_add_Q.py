import re
import sys
from pathlib import Path

_TRAIN_ROOT = Path(__file__).resolve().parent
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))
from _paths import DATA_DIR

INPUT_PATH = DATA_DIR / "raw.txt"
OUTPUT_PATH = DATA_DIR / "marked.txt"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# 用正则切分句子：以 。！？.!? 这些结尾符为分界
pattern = r'([^。！？.!?\n]*[。！？.!?\n])'
sentences = re.findall(pattern, text)

marked_sentences = []
q_count = 0
for s in sentences:
    s_stripped = s.strip()
    if not s_stripped:
        continue
    # 检查是否以问号结尾（中英文都算）
    if s_stripped.endswith("？") or s_stripped.endswith("?"):
        marked_sentences.append("<Q>" + s_stripped)
        q_count += 1
    else:
        marked_sentences.append(s_stripped)

result = "\n".join(marked_sentences)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(result)

print(f"处理完成。总句数：{len(marked_sentences)}，问句数：{q_count}")
print(f"输出文件：{OUTPUT_PATH!s}")








