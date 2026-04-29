"""
由 train/ 里的小 GPT + patterns.json 生成中文问句，供语音对话作主轮刺激音。
路径：始终以 main 的包目录为锚，train 在项目根目录的 train/ 下。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm


def _train_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "train"


def _ensure_train_on_path():
    td = _train_dir()
    s = str(td)
    if s not in sys.path:
        sys.path.insert(0, s)


class QuestionStructureGenerator:
    """封装 generate_Qstructure.py 的逻辑，按需加载权重，可多轮调用 generate_sentence。"""

    def __init__(
        self,
        temperature: float = 0.9,
        top_k: int = 40,
        min_length: int = 12,
        max_length: int = 80,
    ):
        train = _train_dir()
        data_dir = train / "data"
        ckpt_path = train / "checkpoints" / "ckpt.pt"
        tokenizer_path = data_dir / "tokenizer.model"
        patterns_path = data_dir / "patterns.json"

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"找不到分词器: {tokenizer_path}")
        if not patterns_path.is_file():
            raise FileNotFoundError(f"找不到 patterns.json: {patterns_path}\n请先运行 train/finetune/extract_all.py")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

        self.temperature = temperature
        self.top_k = top_k
        self.min_length = min_length
        self.max_length = max_length

        self.punct_wait_steps = 8
        self.question_boost_range = 8
        self.question_boost_strength = 1.5
        self.inject_ranges = {"middle": (0.20, 0.80)}

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(tokenizer_path))

        self.stop_ids = self._build_stop_ids()
        self.punct_ids = self._build_punct_ids()
        sentence_end_ids = self._build_sentence_end_ids()
        q_token_id = self.sp.piece_to_id("<Q>")
        self.forbid_always_ids = set(sentence_end_ids)
        self.forbid_always_ids.add(q_token_id)

        _ensure_train_on_path()
        from model import GPT  # noqa: E402 — 须在加入 train 目录后导入

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        self.model = GPT(cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()

        with open(patterns_path, "r", encoding="utf-8") as f:
            patterns_data = json.load(f)

        self.question_patterns = patterns_data["question_patterns"]
        lengths = patterns_data["sentence_lengths"]
        self.sentence_lengths = [
            l for l in lengths if min_length <= l <= max_length
        ]
        if not self.sentence_lengths:
            self.sentence_lengths = [min(max(l, min_length), max_length) for l in lengths[:50]] or [
                max(min_length, 24)
            ]
        self.start_token_ids = patterns_data["start_token_ids"]

    def _build_stop_ids(self) -> set:
        stop_ids = set()
        for i in range(self.sp.vocab_size()):
            piece = self.sp.id_to_piece(i)
            clean = piece.replace("▁", "")
            if clean in ("？", "?"):
                stop_ids.add(i)
            elif (clean.endswith("？") or clean.endswith("?")) and len(clean) <= 4:
                stop_ids.add(i)
        return stop_ids

    def _build_punct_ids(self) -> set:
        break_chars = set("，；：、,;:")
        punct_ids = set()
        for i in range(self.sp.vocab_size()):
            piece = self.sp.id_to_piece(i)
            clean = piece.replace("▁", "")
            if clean and clean[-1] in break_chars:
                punct_ids.add(i)
        return punct_ids

    def _build_sentence_end_ids(self) -> set:
        sentence_end_chars = set("。！.!")
        s = set()
        for i in range(self.sp.vocab_size()):
            piece = self.sp.id_to_piece(i)
            clean = piece.replace("▁", "")
            if clean and any(c in sentence_end_chars for c in clean):
                s.add(i)
        return s

    def _sample_one_token(
        self,
        idx,
        forbid_ids=None,
        boost_ids=None,
        boost_value=0.0,
        repetition_penalty=1.3,
        repetition_window=20,
        hard_block_consecutive=3,
    ):
        model = self.model
        device = self.device
        temperature = self.temperature
        top_k = self.top_k

        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if repetition_penalty > 1.0 and idx.size(1) > 0:
            recent_window = idx[0, -repetition_window:].tolist()
            recent_set = set(recent_window)
            for tid in recent_set:
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

        if hard_block_consecutive > 0 and idx.size(1) >= hard_block_consecutive:
            last_n = idx[0, -hard_block_consecutive:].tolist()
            if len(set(last_n)) == 1:
                blocked_id = last_n[0]
                logits[:, blocked_id] = float("-inf")

        if forbid_ids:
            for tid in forbid_ids:
                logits[:, tid] = float("-inf")

        if boost_ids and boost_value > 0:
            for tid in boost_ids:
                logits[:, tid] += boost_value

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        return idx_next

    def _maybe_inject_qw(self, idx, qw: str, qw_ids: list[int], check_window: int = 8):
        actual_window = min(check_window, idx.size(1))
        recent_text = self.sp.decode(idx[0, -actual_window:].tolist())
        already_has_qw = qw in recent_text

        if not already_has_qw:
            qw_tensor = torch.tensor([qw_ids], dtype=torch.long, device=self.device)
            idx = torch.cat((idx, qw_tensor), dim=1)

        return idx, not already_has_qw

    def _generate_one_inner(self, pattern, target_length: int):
        sp = self.sp
        stop_ids = self.stop_ids
        punct_ids = self.punct_ids
        forbid_always_ids = self.forbid_always_ids
        max_length = self.max_length
        qbr = self.question_boost_range
        qbs = self.question_boost_strength
        pwd = self.punct_wait_steps
        irr = self.inject_ranges["middle"]

        qw = pattern["question_word"]
        position = pattern["position"]
        temperature = self.temperature
        top_k = self.top_k
        start_token_ids = self.start_token_ids

        device = self.device

        if qw == "<NONE>" or position == "none":
            start_id = random.choice(start_token_ids)
            idx = torch.tensor([[start_id]], dtype=torch.long, device=device)

            while True:
                current_pos = idx.size(1)
                if current_pos >= max_length:
                    break

                forbid_now = set(forbid_always_ids)
                tokens_to_target = target_length - current_pos

                if tokens_to_target <= qbr:
                    boost = (
                        qbs * (qbr - tokens_to_target + 1) / qbr
                    )
                    idx_next = self._sample_one_token(
                        idx,
                        forbid_ids=forbid_now,
                        boost_ids=stop_ids,
                        boost_value=boost,
                    )
                else:
                    forbid_now.update(stop_ids)
                    idx_next = self._sample_one_token(idx, forbid_ids=forbid_now)

                idx = torch.cat((idx, idx_next), dim=1)
                if idx_next.item() in stop_ids:
                    break

            return idx

        if position == "front":
            qw_ids = sp.encode(qw)
            idx = torch.tensor([qw_ids], dtype=torch.long, device=device)

            while True:
                current_pos = idx.size(1)
                if current_pos >= max_length:
                    break

                forbid_now = set(forbid_always_ids)
                tokens_to_target = target_length - current_pos

                if tokens_to_target <= qbr:
                    boost = qbs * (qbr - tokens_to_target + 1) / qbr
                    idx_next = self._sample_one_token(
                        idx,
                        forbid_ids=forbid_now,
                        boost_ids=stop_ids,
                        boost_value=boost,
                    )
                else:
                    forbid_now.update(stop_ids)
                    idx_next = self._sample_one_token(idx, forbid_ids=forbid_now)

                idx = torch.cat((idx, idx_next), dim=1)
                if idx_next.item() in stop_ids:
                    break

            return idx

        if position == "back":
            qw_ids = sp.encode(qw)
            start_id = random.choice(start_token_ids)
            idx = torch.tensor([[start_id]], dtype=torch.long, device=device)

            target_before_qw = max(2, target_length - len(qw_ids) - 1)

            while idx.size(1) < target_before_qw:
                if idx.size(1) >= max_length:
                    break

                forbid_now = set(forbid_always_ids)
                forbid_now.update(stop_ids)

                idx_next = self._sample_one_token(idx, forbid_ids=forbid_now)
                idx = torch.cat((idx, idx_next), dim=1)

            idx, _ = self._maybe_inject_qw(idx, qw, qw_ids)
            forbid_now = set(forbid_always_ids)
            idx_next = self._sample_one_token(
                idx,
                forbid_ids=forbid_now,
                boost_ids=stop_ids,
                boost_value=10.0,
            )
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() not in stop_ids:
                stop_id = next(iter(stop_ids))
                idx = torch.cat((idx, torch.tensor([[stop_id]], dtype=torch.long, device=device)), dim=1)

            return idx

        if position == "middle":
            qw_ids = sp.encode(qw)
            start_id = random.choice(start_token_ids)
            idx = torch.tensor([[start_id]], dtype=torch.long, device=device)

            ratio_lo, ratio_hi = irr
            inject_start = int(target_length * ratio_lo)
            inject_end = int(target_length * ratio_hi)

            qw_injected = False
            punct_wait_count = 0

            while True:
                current_pos = idx.size(1)
                if current_pos >= max_length:
                    break

                if not qw_injected:
                    if current_pos < inject_start:
                        forbid_now = set(forbid_always_ids)
                        forbid_now.update(stop_ids)
                        idx_next = self._sample_one_token(idx, forbid_ids=forbid_now)
                        idx = torch.cat((idx, idx_next), dim=1)

                    elif current_pos < inject_end:
                        forbid_now = set(forbid_always_ids)
                        forbid_now.update(stop_ids)
                        idx_next = self._sample_one_token(idx, forbid_ids=forbid_now)
                        idx = torch.cat((idx, idx_next), dim=1)

                        if idx_next.item() in punct_ids:
                            idx, _ = self._maybe_inject_qw(idx, qw, qw_ids)
                            qw_injected = True

                        else:
                            punct_wait_count += 1
                            if punct_wait_count >= pwd:
                                idx, _ = self._maybe_inject_qw(idx, qw, qw_ids)
                                qw_injected = True

                    else:
                        idx, _ = self._maybe_inject_qw(idx, qw, qw_ids)
                        qw_injected = True

                else:
                    forbid_now = set(forbid_always_ids)
                    tokens_to_target = target_length - current_pos

                    if tokens_to_target <= qbr:
                        boost = qbs * (qbr - tokens_to_target + 1) / qbr
                        idx_next = self._sample_one_token(
                            idx,
                            forbid_ids=forbid_now,
                            boost_ids=stop_ids,
                            boost_value=boost,
                        )
                    else:
                        forbid_now.update(stop_ids)
                        idx_next = self._sample_one_token(idx, forbid_ids=forbid_now)

                    idx = torch.cat((idx, idx_next), dim=1)
                    if idx_next.item() in stop_ids:
                        break

            return idx

        raise ValueError(f"未知的 position: {position}")

    def generate_sentence(self) -> str:
        """采样模式与目标长度，生成一句问话（解码为字符串）。"""
        pattern = random.choice(self.question_patterns)
        target_length = random.choice(self.sentence_lengths)

        out = self._generate_one_inner(pattern, target_length)
        text = self.sp.decode(out[0].tolist()).strip()
        if not (text.endswith("？") or text.endswith("?")):
            text += "？"
        return text
