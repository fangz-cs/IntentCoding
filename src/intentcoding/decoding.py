"""IntentCoding token ensemble and cache-aware beam search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from intentcoding.masking import PromptViews

DEFAULT_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class DecodingConfig:
    """Parameters from the ACL 2026 IntentCoding experiments."""

    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    beam_size: int = 4
    max_new_tokens: int = 512
    num_return_sequences: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "alphas", tuple(self.alphas))
        if not self.alphas:
            raise ValueError("at least one amplification strength is required")
        if not all(math.isfinite(alpha) for alpha in self.alphas):
            raise ValueError("amplification strengths must be finite")
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not 1 <= self.num_return_sequences <= self.beam_size:
            raise ValueError(
                "num_return_sequences must be between 1 and beam_size"
            )


@dataclass(frozen=True)
class TokenCandidate:
    """One unique top-1 token and its ensemble probability."""

    token_id: int
    probability: float


@dataclass(frozen=True)
class Generation:
    """A ranked generation returned by IntentCoding."""

    token_ids: tuple[int, ...]
    score: float
    mean_log_probability: float
    finished: bool
    steps: int


@dataclass(frozen=True)
class _Beam:
    token_ids: tuple[int, ...]
    score: float
    steps: int


@dataclass(frozen=True)
class _Child:
    beam: _Beam
    parent_index: int


def ensemble_candidates(
    original_logits: Tensor,
    masked_logits: Tensor,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> tuple[TokenCandidate, ...]:
    """Apply multi-strength intent amplification and group top-1 tokens.

    For each strength ``alpha``, this computes
    ``original + alpha * (original - masked)``. If multiple strengths choose
    the same top-1 token, its softmax probabilities are averaged as described
    by Equation 4 in the paper.
    """

    if original_logits.ndim != 1 or masked_logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if original_logits.shape != masked_logits.shape:
        raise ValueError("original and masked logits must have the same shape")
    if original_logits.numel() == 0:
        raise ValueError("logits must not be empty")
    if not alphas:
        raise ValueError("at least one amplification strength is required")

    intent_signal = original_logits - masked_logits
    grouped: dict[int, list[float]] = {}
    for alpha in alphas:
        if not math.isfinite(float(alpha)):
            raise ValueError("amplification strengths must be finite")
        amplified = original_logits + float(alpha) * intent_signal
        probabilities = torch.softmax(amplified.float(), dim=-1)
        token_id = int(torch.argmax(probabilities).item())
        grouped.setdefault(token_id, []).append(
            float(probabilities[token_id].item())
        )

    return tuple(
        TokenCandidate(
            token_id=token_id,
            probability=sum(probabilities) / len(probabilities),
        )
        for token_id, probabilities in grouped.items()
    )


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device(getattr(model, "device", "cpu"))


def _require_cache(cache: Any) -> Any:
    if cache is None:
        raise RuntimeError("the model did not return a key-value cache")
    return cache


def _select_cache(
    cache: Any,
    batch_indices: Sequence[int],
    device: torch.device,
) -> Any:
    """Select or duplicate cache rows across Transformers cache versions."""

    cache = _require_cache(cache)
    indices = torch.tensor(batch_indices, dtype=torch.long, device=device)

    selector = getattr(cache, "batch_select_indices", None)
    if callable(selector):
        selector(indices)
        return cache

    reorder = getattr(cache, "reorder_cache", None)
    if callable(reorder):
        reorder(indices)
        return cache

    converter = getattr(cache, "to_legacy_cache", None)
    if callable(converter):
        cache = converter()
    if not isinstance(cache, (tuple, list)):
        raise TypeError("unsupported key-value cache representation")

    selected_layers = []
    for layer in cache:
        if not isinstance(layer, (tuple, list)):
            raise TypeError("unsupported key-value cache layer")
        selected_states = []
        for state in layer:
            if not isinstance(state, Tensor):
                raise TypeError("key-value cache states must be tensors")
            selected_states.append(
                state.index_select(0, indices.to(state.device))
            )
        selected_layers.append(tuple(selected_states))
    return tuple(selected_layers)


def _log_probability(probability: float) -> float:
    if not 0.0 < probability <= 1.0:
        raise ValueError("candidate probability must be in (0, 1]")
    return math.log(probability)


def _rank_beams(beams: Sequence[_Beam]) -> list[_Beam]:
    return sorted(beams, key=lambda beam: (-beam.score, beam.token_ids))


def _as_generation(beam: _Beam, finished: bool) -> Generation:
    return Generation(
        token_ids=beam.token_ids,
        score=beam.score,
        mean_log_probability=beam.score / beam.steps,
        finished=finished,
        steps=beam.steps,
    )


class IntentDecoder:
    """Cache-aware IntentCoding decoder for causal language models."""

    def __init__(
        self,
        model: Any,
        config: DecodingConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or DecodingConfig()

    @torch.inference_mode()
    def generate(
        self,
        views: PromptViews,
        eos_token_ids: int | Sequence[int],
    ) -> list[Generation]:
        """Generate and rank code hypotheses from one prompt."""

        eos_ids = (
            {int(eos_token_ids)}
            if isinstance(eos_token_ids, int)
            else {int(token_id) for token_id in eos_token_ids}
        )
        if not eos_ids:
            raise ValueError("at least one EOS token ID is required")

        device = _model_device(self.model)
        views = views.to(device)
        paired_input_ids = views.input_ids.repeat(2, 1)
        paired_attention = torch.cat(
            [views.attention_mask, views.masked_attention_mask],
            dim=0,
        )

        output = self.model(
            input_ids=paired_input_ids,
            attention_mask=paired_attention,
            use_cache=True,
            return_dict=True,
        )
        prefix_cache = _require_cache(output.past_key_values)
        candidates = ensemble_candidates(
            output.logits[0, -1, :],
            output.logits[1, -1, :],
            self.config.alphas,
        )

        active: list[_Beam] = []
        completed: list[Generation] = []
        for candidate in candidates:
            score = _log_probability(candidate.probability)
            if candidate.token_id in eos_ids:
                completed.append(
                    Generation(
                        token_ids=(),
                        score=score,
                        mean_log_probability=score,
                        finished=True,
                        steps=1,
                    )
                )
            else:
                active.append(
                    _Beam(
                        token_ids=(candidate.token_id,),
                        score=score,
                        steps=1,
                    )
                )

        active = _rank_beams(active)[: self.config.beam_size]
        if not active:
            return self._finalize(completed)

        cache = _select_cache(
            prefix_cache,
            [index for _ in active for index in (0, 1)],
            device,
        )
        last_token_ids = self._paired_last_tokens(active, device)
        attention_mask = torch.cat(
            [paired_attention for _ in active],
            dim=0,
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    attention_mask.shape[0],
                    1,
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

        generated_length = 1
        while active and generated_length < self.config.max_new_tokens:
            output = self.model(
                input_ids=last_token_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            next_cache = _require_cache(output.past_key_values)
            children: list[_Child] = []

            for parent_index, parent in enumerate(active):
                pair_start = parent_index * 2
                candidates = ensemble_candidates(
                    output.logits[pair_start, -1, :],
                    output.logits[pair_start + 1, -1, :],
                    self.config.alphas,
                )
                for candidate in candidates:
                    score = parent.score + _log_probability(
                        candidate.probability
                    )
                    steps = parent.steps + 1
                    if candidate.token_id in eos_ids:
                        completed.append(
                            Generation(
                                token_ids=parent.token_ids,
                                score=score,
                                mean_log_probability=score / steps,
                                finished=True,
                                steps=steps,
                            )
                        )
                        continue

                    children.append(
                        _Child(
                            beam=_Beam(
                                token_ids=(
                                    *parent.token_ids,
                                    candidate.token_id,
                                ),
                                score=score,
                                steps=steps,
                            ),
                            parent_index=parent_index,
                        )
                    )

            children.sort(
                key=lambda child: (
                    -child.beam.score,
                    child.beam.token_ids,
                )
            )
            children = children[: self.config.beam_size]
            if not children:
                active = []
                break

            cache_indices = [
                index
                for child in children
                for index in (
                    child.parent_index * 2,
                    child.parent_index * 2 + 1,
                )
            ]
            cache = _select_cache(next_cache, cache_indices, device)
            parent_attention = torch.cat(
                [
                    attention_mask[
                        child.parent_index * 2 :
                        child.parent_index * 2 + 2
                    ]
                    for child in children
                ],
                dim=0,
            )
            attention_mask = torch.cat(
                [
                    parent_attention,
                    torch.ones(
                        parent_attention.shape[0],
                        1,
                        dtype=parent_attention.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            active = [child.beam for child in children]
            last_token_ids = self._paired_last_tokens(active, device)
            generated_length += 1

        completed.extend(_as_generation(beam, finished=False) for beam in active)
        return self._finalize(completed)

    def _finalize(
        self,
        generations: Sequence[Generation],
    ) -> list[Generation]:
        ranked = sorted(
            generations,
            key=lambda item: (
                not item.finished,
                -item.mean_log_probability,
                -item.score,
                item.token_ids,
            ),
        )
        return ranked[: self.config.num_return_sequences]

    @staticmethod
    def _paired_last_tokens(
        beams: Sequence[_Beam],
        device: torch.device,
    ) -> Tensor:
        return torch.tensor(
            [
                token_id
                for beam in beams
                for token_id in (
                    beam.token_ids[-1],
                    beam.token_ids[-1],
                )
            ],
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
