"""
命令行：基于问句结构与句长分布，生成多条中文问句（逻辑由 question_generator 提供）。
项目根路径：python main/generate_Qstructure.py 须在仓库根目录下运行，或由 IDE 设定工作目录包含 main。

等价于此前独立脚本：训练数据与 ckpt 在 ../train/ 。
"""
from __future__ import annotations

import argparse

from question_generator import QuestionStructureGenerator


def main():
    parser = argparse.ArgumentParser(description="由 train GPT 批量生成中文问句")
    parser.add_argument("-n", "--num", type=int, default=1, help="生成条数（默认 1）")
    args = parser.parse_args()

    print("初始化生成器...")
    gen = QuestionStructureGenerator()

    print(f"\n生成 {args.num} 条问句\n" + "=" * 60)

    for i in range(args.num):
        text = gen.generate_sentence()
        print(f"[{i + 1:2d}] {text}\n")

    print("=" * 60 + "\n完成")


if __name__ == "__main__":
    main()
