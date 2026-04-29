"""train 包根目录与数据、checkpoint 路径（相对本文件，不依赖当前工作目录）"""
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parent
DATA_DIR = TRAIN_ROOT / "data"
CHECKPOINTS_DIR = TRAIN_ROOT / "checkpoints"
