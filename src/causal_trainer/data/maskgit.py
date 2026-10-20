"""Dynamic, packing-safe MaskGIT augmentation of ChatML image responses.

Runs on host batches, after all existing preprocessing. Prepared artifacts stay
unchanged; only three O(B*S) arrays are added to batches when enabled. All random
choices depend on the seed and absolute optimizer step, including after resume.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from itertools import pairwise

import numpy as np

_IMAGE = re.compile(r"<\|image_[0-9]+\|>\Z")


class MaskGITBatchTransform:
    """Select each complete, supervised assistant image block with probability p.

    Selection is per image, independently of its token masking ratio. Responses
    missing their closing message delimiter (e.g. truncation) remain ordinary AR.
    Grid boundaries and shape come from assistant tokens, never the user prompt.
    Input images and ordinary messages are untouched. Existing weights are honored.
    """

    def __init__(self, tokenizer, *, probability: float, seed: int) -> None:
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("maskgit_probability must be between 0 and 1")
        self.probability = probability
        self.seed = seed
        self.tokenizer = tokenizer
        vocabulary = tokenizer.get_vocab()
        required = ("[MASK]", "<|im_start|>", "<|im_end|>")
        for token in required:
            if token not in vocabulary or tokenizer(token)["input_ids"] != [vocabulary[token]]:
                raise ValueError(f"MaskGIT requires {token!r} to encode as one token")
        self.mask_id, self.start_id, self.end_id = (vocabulary[token] for token in required)
        image_ids = [value for token, value in vocabulary.items() if _IMAGE.fullmatch(token)]
        if not image_ids:
            raise ValueError("MaskGIT requires <|image_i|> tokens in the tokenizer vocabulary")
        max_id = max(max(vocabulary.values()), getattr(tokenizer, "pad_token_id", None) or 0)
        self.is_image = np.zeros(max_id + 1, dtype=np.bool_)
        self.is_image[image_ids] = True

    def _blocks(self, tokens, valid, segments):
        """Yield actual image grids inside complete assistant messages."""
        if not self.is_image[tokens].any():
            return
        boundaries = np.flatnonzero(
            np.r_[True, (segments[1:] != segments[:-1]) | (valid[1:] != valid[:-1]), True]
        )
        for left, right in pairwise(boundaries):
            if not valid[left]:
                continue
            starts = left + np.flatnonzero(tokens[left:right] == self.start_id)
            ends = left + np.flatnonzero(tokens[left:right] == self.end_id)
            for index, start in enumerate(starts):
                next_start = starts[index + 1] if index + 1 < len(starts) else right
                end_index = np.searchsorted(ends, start)
                if end_index == len(ends) or ends[end_index] >= next_start:
                    continue
                end = int(ends[end_index])
                header = self.tokenizer.decode(tokens[start + 1:min(end, start + 33)].tolist())
                if not header.startswith(("assistant\n", "assistant\r\n")):
                    continue
                images = start + 1 + np.flatnonzero(self.is_image[tokens[start + 1:end]])
                if not len(images):
                    continue
                # Consecutive codes form rows. A single newline joins rows;
                # other text (including a blank line) separates image blocks.
                gaps = np.flatnonzero(np.diff(images) > 1)
                splits = []
                row_breaks = []
                for i in gaps:
                    between = tokens[images[i] + 1:images[i + 1]]
                    gap = self.tokenizer.decode(between[:8].tolist()) if len(between) <= 8 else None
                    if gap in ("\n", "\r\n"):
                        row_breaks.append(int(i) + 1)
                    else:
                        splits.append(int(i) + 1)
                stops = [0, *splits, len(images)]
                for first, last in pairwise(stops):
                    rows = [first, *(i for i in row_breaks if first < i < last), last]
                    widths = np.diff(rows)
                    if np.any(widths != widths[0]):
                        raise ValueError("assistant image grid has unequal row widths")
                    block = images[first:last]
                    yield int(block[0]), int(block[-1]) + 1, block

    def __call__(self, batch: Mapping[str, np.ndarray], *, step: int) -> dict[str, np.ndarray]:
        if self.probability == 0.0:
            return dict(batch)
        original = np.asarray(batch["input_ids"], dtype=np.int32)
        valid = np.asarray(batch["attention_mask"]) != 0
        segments = np.asarray(batch["segment_ids"])
        weights = np.asarray(batch["loss_weights"], dtype=np.float32)
        tokens = original.copy()
        targets = np.zeros_like(original)
        prediction_weights = np.zeros_like(weights)
        targets[:, :-1] = original[:, 1:]
        edges = valid[:, :-1] & valid[:, 1:] & (segments[:, :-1] == segments[:, 1:])
        prediction_weights[:, :-1] = weights[:, 1:] * edges
        attention_end = np.broadcast_to(np.arange(original.shape[1], dtype=np.int32), original.shape).copy()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        for row in range(len(original)):
            for start, end, images in self._blocks(original[row], valid[row], segments[row]):
                # Disabled turns/dummy rows must not introduce corrupt context.
                if not np.any(weights[row, images] > 0) or rng.random() >= self.probability:
                    continue
                count = max(1, math.ceil(len(images) * math.cos(math.pi * 0.5 * rng.random())))
                masked = rng.choice(images, size=count, replace=False)
                tokens[row, masked] = self.mask_id
                attention_end[row, start:end] = end - 1
                # Remove every AR edge into/from this block, including newline
                # and closing-delimiter targets. They are supplied by the template.
                prediction_weights[row, max(0, start - 1):end] = 0.0
                targets[row, masked] = original[row, masked]
                prediction_weights[row, masked] = weights[row, masked]
        return {
            **batch,
            "input_ids": tokens,
            "attention_end": attention_end,
            "prediction_targets": targets,
            "prediction_weights": prediction_weights,
        }

    def batches(self, batches, *, start_step: int) -> Iterator:
        for step, batch in enumerate(batches, start=start_step):
            if isinstance(batch, Mapping):
                yield self(batch, step=step)
            else:
                yield batch.with_data(self(batch.data, step=step))
