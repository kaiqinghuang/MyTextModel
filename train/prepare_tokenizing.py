import sys
from pathlib import Path
import sentencepiece as spm

_TRAIN_ROOT = Path(__file__).resolve().parent
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))
from _paths import DATA_DIR

INPUT_PATH = str(DATA_DIR / "marked.txt")
MODEL_PREFIX = str(DATA_DIR / "tokenizer")

spm.SentencePieceTrainer.train(
    input=INPUT_PATH,
    model_prefix=MODEL_PREFIX,
    vocab_size=5000,
    model_type="bpe",
    character_coverage=0.9995,
    user_defined_symbols=["<Q>"],   # 把<Q>注册为单独的token
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)

print(f"分词器训练完成，输出：{MODEL_PREFIX}.model 和 {MODEL_PREFIX}.vocab")

# 验证：加载分词器，编码几个样本
sp = spm.SentencePieceProcessor()
sp.load(MODEL_PREFIX + ".model")

print(f"\n词表大小：{sp.vocab_size()}")
print(f"<Q>的token id：{sp.piece_to_id('<Q>')}")

test_strings = [
    "<Q>你今天过得怎么样？",
    "这是一个测试句子。",
]
for s in test_strings:
    ids = sp.encode(s)
    pieces = sp.encode(s, out_type=str)
    print(f"\n原文：{s}")
    print(f"分词：{pieces}")
    print(f"token ids：{ids}")