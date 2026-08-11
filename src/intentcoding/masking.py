"""Build the original and intent-masked attention views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

MASK_MARKER = "<mask_ins>"


@dataclass(frozen=True)
class PromptViews:
    """Token IDs and attention masks for the two IntentCoding views."""

    input_ids: Tensor
    attention_mask: Tensor
    masked_attention_mask: Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape [1, sequence_length]")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if self.masked_attention_mask.shape != self.input_ids.shape:
            raise ValueError("masked_attention_mask must match input_ids")

    def to(self, device: torch.device | str) -> "PromptViews":
        """Move all tensors to a device."""

        return PromptViews(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            masked_attention_mask=self.masked_attention_mask.to(device),
        )


def _validate_masked_prompt(
    prompt: str,
    masked_prompt: str,
    marker: str,
) -> tuple[int, int]:
    if not marker:
        raise ValueError("mask marker must not be empty")
    if masked_prompt.count(marker) != 1:
        raise ValueError(
            f"masked prompt must contain exactly one {marker!r} marker"
        )

    prefix, suffix = masked_prompt.split(marker)
    if not prompt.startswith(prefix):
        raise ValueError("text before the mask marker differs from the prompt")
    if suffix and not prompt.endswith(suffix):
        raise ValueError("text after the mask marker differs from the prompt")
    if len(prompt) < len(prefix) + len(suffix):
        raise ValueError("masked prompt does not describe a span in the prompt")
    span_start = len(prefix)
    span_end = len(prompt) - len(suffix) if suffix else len(prompt)
    if span_start >= span_end:
        raise ValueError("masked intent span must not be empty")
    return span_start, span_end


def _ensure_marker_token(tokenizer: Any, marker: str) -> int:
    vocab = tokenizer.get_vocab()
    if marker not in vocab:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [marker]}
        )

    marker_id = tokenizer.convert_tokens_to_ids(marker)
    if marker_id is None:
        raise ValueError(f"tokenizer could not register mask marker {marker!r}")
    return int(marker_id)


def encode_prompt_views(
    tokenizer: Any,
    prompt: str,
    masked_prompt: str,
    marker: str = MASK_MARKER,
) -> PromptViews:
    """Tokenize a prompt and disable attention over its masked intent span.

    ``masked_prompt`` must equal ``prompt`` except that one contiguous span is
    replaced by ``marker``. The masked prompt IDs are used only to locate that
    span; the language model receives the original IDs in both batch rows.
    """

    character_start, character_end = _validate_masked_prompt(
        prompt,
        masked_prompt,
        marker,
    )

    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        overlaps_span = (
            (offsets[:, :, 0] < character_end)
            & (offsets[:, :, 1] > character_start)
        )
        if not bool(overlaps_span.any()):
            raise ValueError(
                "masked span could not be aligned after tokenization"
            )
        masked_attention_mask = attention_mask.clone()
        masked_attention_mask[overlaps_span] = 0
        return PromptViews(
            input_ids=input_ids,
            attention_mask=attention_mask,
            masked_attention_mask=masked_attention_mask,
        )

    marker_id = _ensure_marker_token(tokenizer, marker)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded_masked = tokenizer(masked_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    masked_ids = encoded_masked["input_ids"]

    marker_positions = torch.nonzero(
        masked_ids[0] == marker_id,
        as_tuple=False,
    ).flatten()
    if marker_positions.numel() != 1:
        raise ValueError("mask marker must tokenize to one unique token")

    span_start = int(marker_positions.item())
    span_length = input_ids.shape[1] - masked_ids.shape[1] + 1
    span_end = span_start + span_length
    if span_length <= 0 or span_end > input_ids.shape[1]:
        raise ValueError("masked span could not be aligned after tokenization")

    masked_attention_mask = attention_mask.clone()
    masked_attention_mask[:, span_start:span_end] = 0
    return PromptViews(
        input_ids=input_ids,
        attention_mask=attention_mask,
        masked_attention_mask=masked_attention_mask,
    )
