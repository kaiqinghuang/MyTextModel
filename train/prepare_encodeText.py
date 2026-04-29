import sys
from pathlib import Path
import numpy as np
import sentencepiece as spm

_TRAIN_ROOT = Path(__file__).resolve().parent
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))
from _paths import DATA_DIR

INPUT_PATH = DATA_DIR / "marked.txt"
TOKENIZER_PATH = DATA_DIR / "tokenizer.model"
OUTPUT_PATH = DATA_DIR / "train.bin"

sp = spm.SentencePieceProcessor()
sp.load(str(TOKENIZER_PATH))

with open(str(INPUT_PATH), "r", encoding="utf-8") as f:
    text = f.read()

ids = sp.encode(text)
print(f"原文字符数：{len(text)}")
print(f"编码后token数：{len(ids)}")
print(f"压缩比（字符/token）：{len(text)/len(ids):.2f}")

max_id = max(ids)
print(f"最大token id：{max_id}（uint16上限65535）")
assert max_id < 65535, "token id超过uint16范围"

ids_array = np.array(ids, dtype=np.uint16)
ids_array.tofile(str(OUTPUT_PATH))

print(f"\n保存到：{OUTPUT_PATH!s}")
print(f"文件大小：{ids_array.nbytes/1024:.2f} KB")