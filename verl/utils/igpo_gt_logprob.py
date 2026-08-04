"""IGPO (Information Gain-based Policy Optimization, arXiv:2510.14967) ground-truth
log-probability scoring and turn-level information-gain reward construction.

Design
------
IGPO turns each search turn into an "information acquisition" step. After turn t the
policy's belief in the correct answer is measured by P(GT | history_t), scored via
*teacher forcing* (not generation): we append the ground-truth answer to the history
and read off the mean log-probability of the GT answer tokens. The per-turn reward is
the marginal gain of that belief:
    prob_diff  (default):  r_t = exp(mean_logP_t) - exp(mean_logP_{t-1})
    log_prob_diff       :  r_t = mean_logP_t       - mean_logP_{t-1}

This module is split so that everything except the actual forward pass is pure and
CPU-testable:
  * tokenize_ground_truth / get_answer_token_range  -> tokenization (needs a tokenizer, no GPU)
  * build_gt_scoring_rows                            -> builds [history_t + GT] rows + bookkeeping
  * compute_info_gain_per_turn                       -> the prob_diff/log_prob_diff formula (pure)
  * place_info_gain_on_tokens                        -> scatter per-turn rewards onto token positions (pure)
  * IGPORewardBuilder.build                          -> orchestration; the GPU forward is injected
                                                        as a callable `logprob_fn`, so tests pass a fake.
The trainer (ray_trainer) wires `logprob_fn = actor_rollout_wg.compute_log_prob`.

NOTE
----
The original source of this module was lost (see report/); this file is a faithful
reconstruction from (1) the compiled ``igpo_gt_logprob.cpython-310.pyc`` symbol table and
docstrings, (2) the caller contract in ``verl/trainer/ppo/ray_trainer.py`` (``_compute_igpo_info_gain``
and ``_igpo_score_rows``), and (3) the official reference implementation under
``report/IGPO_ref`` (``scrl/llm_agent/vectorized_gt_logprob.py`` and ``generation.py``).
The external interface exactly matches what the trainer invokes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch

__all__ = [
    "tokenize_ground_truth",
    "get_answer_token_range",
    "compute_info_gain_per_turn",
    "place_info_gain_on_tokens",
    "GTScoringRow",
    "build_gt_scoring_rows",
    "IGPORewardBuilder",
]


# ---------------------------------------------------------------------------
# Tokenization (needs a tokenizer, no GPU)
# ---------------------------------------------------------------------------
def tokenize_ground_truth(gt_text: str, prefix: str, suffix: str, tokenizer) -> list[int]:
    """Tokenize PREFIX + gt_text + SUFFIX into ids (no special tokens added)."""
    full_text = f"{prefix}{gt_text}{suffix}"
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    return list(input_ids)


def get_answer_token_range(gt_text: str, prefix: str, suffix: str, tokenizer) -> tuple[int, int]:
    """Return the [start, end) token range of the *answer* portion inside PREFIX+gt+SUFFIX.

    Uses offset_mapping for precise char->token boundary detection, mirroring the official
    ``VectorizedGTLogProbComputer.get_gt_answer_token_range``. The tokenization here uses
    ``add_special_tokens=False`` so the returned indices line up with the ids produced by
    :func:`tokenize_ground_truth` (which the scoring rows are built from).
    """
    full_text = f"{prefix}{gt_text}{suffix}"
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]

    gt_char_start = len(prefix)
    gt_char_end = len(prefix) + len(gt_text)

    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, (char_start, char_end) in enumerate(offsets):
        if start_idx is None and char_end > gt_char_start:
            start_idx = i
        if char_start < gt_char_end and char_end > 0:
            end_idx = i + 1

    if start_idx is None:
        start_idx = len(offsets)
    if end_idx is None:
        end_idx = len(offsets)
    return start_idx, end_idx


# ---------------------------------------------------------------------------
# Info-gain formula (pure)
# ---------------------------------------------------------------------------
def compute_info_gain_per_turn(turn_mean_logprobs: list[float], info_gain_type: str = "prob_diff") -> list[float]:
    """Given per-turn mean log-prob of the GT answer [logP_0, logP_1, ..., logP_{T-1}],
    return the per-turn info-gain reward for turns 1..T-1 (turn 0 seeds the baseline, no reward).
    Mirrors official generation.py:558-610.
      prob_diff:     r_t = exp(logP_t) - exp(logP_{t-1})
      log_prob_diff: r_t = logP_t - logP_{t-1}
    NaN/inf-safe: any invalid step contributes 0.0.
    """

    def to_value(lp: Optional[float]) -> Optional[float]:
        if lp is None:
            return None
        if math.isnan(lp) or math.isinf(lp):
            return None
        if info_gain_type == "log_prob_diff":
            return lp
        elif info_gain_type == "prob_diff":
            return math.exp(lp)
        else:
            raise ValueError(f"unknown info_gain_type: {info_gain_type}")

    gains: list[float] = []
    if not turn_mean_logprobs:
        return gains

    prev = to_value(turn_mean_logprobs[0])
    for t in range(1, len(turn_mean_logprobs)):
        curr = to_value(turn_mean_logprobs[t])
        if prev is None or curr is None:
            gains.append(0.0)
        else:
            g = curr - prev
            gains.append(0.0 if (math.isnan(g) or math.isinf(g)) else g)
        if curr is not None:
            prev = curr
    return gains


# ---------------------------------------------------------------------------
# Reward placement (pure)
# ---------------------------------------------------------------------------
def place_info_gain_on_tokens(
    info_gains: list[float],
    turn_end_indices: list[int],
    response_length: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter turn-level info-gain rewards onto the last token of each turn.

    ``info_gains[k]`` is the gain for turn (k+1) relative to turn k (see
    :func:`compute_info_gain_per_turn`), so it is placed at the boundary of turn (k+1),
    i.e. ``turn_end_indices[k+1] - 1``.

    Returns:
        reward_row:        (response_length,) info-gain reward per token (0 except turn ends)
        turn_boundary_row: (response_length,) 1 at every turn-end position (marks turn structure
                           for the turn-level advantage estimator), 0 elsewhere.
    """
    reward_row = torch.zeros(response_length, dtype=dtype)
    turn_boundary_row = torch.zeros(response_length, dtype=dtype)

    if response_length <= 0:
        return reward_row, turn_boundary_row

    # Mark every turn-end position (position of the turn's last response token = e - 1).
    for e in turn_end_indices:
        pos = min(max(int(e) - 1, 0), response_length - 1)
        turn_boundary_row[pos] = 1.0

    # Place info_gains[k] at the end of turn (k+1).
    for k, g in enumerate(info_gains):
        boundary_turn = turn_end_indices[k + 1]
        pos = min(max(int(boundary_turn) - 1, 0), response_length - 1)
        reward_row[pos] = float(g)

    return reward_row, turn_boundary_row


# ---------------------------------------------------------------------------
# Teacher-forcing rows
# ---------------------------------------------------------------------------
@dataclass
class GTScoringRow:
    """One teacher-forcing row: [prompt + response[:turn_end] + GT]."""

    sample_idx: int
    turn_idx: int
    input_ids: list[int]
    ans_start: int
    ans_end: int


def _usable_boundaries(response_len: int, boundaries, response_length: int) -> list[int]:
    """Filter/clamp per-sample turn-end boundaries to the ones that are actually scorable.

    A boundary ``e`` is the number of response tokens up to and including a turn end; its
    last-token position is ``e - 1``. Boundaries beyond ``response_length`` (truncated turns)
    or beyond the sample's real response length are dropped. Order is preserved and duplicates
    are collapsed.
    """
    cap = min(int(response_len), int(response_length))
    out: list[int] = []
    for e in boundaries or []:
        e = int(e)
        if 1 <= e <= cap and (not out or out[-1] != e):
            out.append(e)
    return out


def build_gt_scoring_rows(
    prompt_ids_per_sample: list[list[int]],
    response_ids_per_sample: list[list[int]],
    turn_end_indices_per_sample: list[list[int]],
    gt_tokens_per_sample: list[list[int]],
    gt_answer_range_per_sample: list[tuple[int, int]],
    response_length: int,
) -> list[GTScoringRow]:
    """Build one teacher-forcing row per (sample, turn boundary).

    For sample i and turn boundary e (index into the response tokens), the row is:
        prompt_i + response_i[:e] + gt_tokens_i
    and the GT answer tokens sit at [len(prompt_i)+e + ans_start, ... + ans_end).
    Boundaries beyond ``response_length`` are skipped (those turns were truncated downstream).
    Samples with no GT or <2 usable turns yield no rows (no info gain possible).
    """
    rows: list[GTScoringRow] = []
    bsz = len(prompt_ids_per_sample)
    for i in range(bsz):
        gt_tokens = gt_tokens_per_sample[i]
        ans_start_in_gt, ans_end_in_gt = gt_answer_range_per_sample[i]
        if not gt_tokens or ans_end_in_gt <= ans_start_in_gt:
            continue

        prompt = prompt_ids_per_sample[i]
        response = response_ids_per_sample[i]
        boundaries = _usable_boundaries(len(response), turn_end_indices_per_sample[i], response_length)
        if len(boundaries) < 2:
            # Need at least two beliefs to form one info-gain step.
            continue

        for turn_idx, e in enumerate(boundaries):
            history = prompt + response[:e]
            row_ids = history + list(gt_tokens)
            base = len(history)
            rows.append(
                GTScoringRow(
                    sample_idx=i,
                    turn_idx=turn_idx,
                    input_ids=row_ids,
                    ans_start=base + ans_start_in_gt,
                    ans_end=base + ans_end_in_gt,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
class IGPORewardBuilder:
    """Builds token-level IGPO info-gain rewards for a rollout batch.

    The only GPU-dependent step (teacher-forcing forward) is injected as ``logprob_fn``:
        logprob_fn(rows: list[GTScoringRow]) -> list[float]
    returning, for each row, the mean log-prob of that row's GT answer tokens. Tests pass a
    deterministic fake; the trainer passes a wrapper around actor.compute_log_prob.
    """

    def __init__(
        self,
        tokenizer,
        info_gain_type: str = "prob_diff",
        gt_prefix: str = "\nNow there's enough information to answer\n</thought>\n<answer>\n",
        gt_suffix: str = "\n</answer><|im_end|>",
    ):
        self.tokenizer = tokenizer
        self.info_gain_type = info_gain_type
        self.gt_prefix = gt_prefix
        self.gt_suffix = gt_suffix

    def _tokenize_gts(self, gt_texts: list[str]):
        gt_tokens_per_sample: list[list[int]] = []
        gt_range_per_sample: list[tuple[int, int]] = []
        for gt in gt_texts:
            if not gt:
                gt_tokens_per_sample.append([])
                gt_range_per_sample.append((0, 0))
                continue
            gt_tokens_per_sample.append(
                tokenize_ground_truth(gt, self.gt_prefix, self.gt_suffix, self.tokenizer)
            )
            gt_range_per_sample.append(
                get_answer_token_range(gt, self.gt_prefix, self.gt_suffix, self.tokenizer)
            )
        return gt_tokens_per_sample, gt_range_per_sample

    def build(
        self,
        prompt_ids_per_sample: list[list[int]],
        response_ids_per_sample: list[list[int]],
        turn_end_indices_per_sample: list[list[int]],
        gt_texts: list[str],
        response_length: int,
        logprob_fn: Callable[[list[GTScoringRow]], list[float]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (info_gain_reward, turn_boundary_mask), each shape (bsz, response_length).

        Rows for all (sample, turn) pairs are scored in ONE logprob_fn call (batched), then
        the per-turn mean log-probs are turned into info-gain and scattered onto token ends.
        """
        bsz = len(prompt_ids_per_sample)
        info_gain_reward = torch.zeros(bsz, response_length, dtype=torch.float32)
        turn_boundary_mask = torch.zeros(bsz, response_length, dtype=torch.float32)

        gt_tokens_per_sample, gt_range_per_sample = self._tokenize_gts(gt_texts)
        rows = build_gt_scoring_rows(
            prompt_ids_per_sample,
            response_ids_per_sample,
            turn_end_indices_per_sample,
            gt_tokens_per_sample,
            gt_range_per_sample,
            response_length,
        )
        if not rows:
            return info_gain_reward, turn_boundary_mask

        mean_logprobs = logprob_fn(rows)
        if len(mean_logprobs) != len(rows):
            raise ValueError("logprob_fn must return one value per row")

        # Group per-turn mean log-probs by sample, preserving row (turn) order.
        per_sample_logps: dict[int, list[float]] = {}
        for row, lp in zip(rows, mean_logprobs):
            per_sample_logps.setdefault(row.sample_idx, []).append(float(lp))

        for i, mean_logps in per_sample_logps.items():
            if len(mean_logps) < 2:
                continue
            # Recompute the exact usable boundaries used to build this sample's rows so the
            # gains line up with the boundary positions.
            boundaries = _usable_boundaries(
                len(response_ids_per_sample[i]), turn_end_indices_per_sample[i], response_length
            )
            info_gains = compute_info_gain_per_turn(mean_logps, self.info_gain_type)
            r_row, b_row = place_info_gain_on_tokens(
                info_gains, boundaries, response_length, torch.float32
            )
            info_gain_reward[i] = r_row
            turn_boundary_mask[i] = b_row

        return info_gain_reward, turn_boundary_mask
