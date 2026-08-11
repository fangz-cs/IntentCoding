from types import SimpleNamespace

import pytest
import torch

from intentcoding.decoding import (
    DEFAULT_ALPHAS,
    DecodingConfig,
    IntentDecoder,
    ensemble_candidates,
)
from intentcoding.masking import PromptViews


class ConstantModel:
    def __init__(self, token_id: int, vocab_size: int = 4) -> None:
        self.device = torch.device("cpu")
        self.token_id = token_id
        self.vocab_size = vocab_size
        self.calls = 0

    def parameters(self):
        return iter(())

    def __call__(
        self,
        input_ids,
        attention_mask,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ):
        del attention_mask, past_key_values, use_cache, return_dict
        self.calls += 1
        batch_size, sequence_length = input_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self.vocab_size),
            -10.0,
        )
        logits[:, :, self.token_id] = 10.0
        state = torch.arange(batch_size, dtype=torch.float32).reshape(
            batch_size,
            1,
            1,
            1,
        )
        return SimpleNamespace(
            logits=logits,
            past_key_values=((state, state.clone()),),
        )


def prompt_views() -> PromptViews:
    return PromptViews(
        input_ids=torch.tensor([[5, 6]]),
        attention_mask=torch.tensor([[1, 1]]),
        masked_attention_mask=torch.tensor([[0, 1]]),
    )


def test_paper_defaults() -> None:
    config = DecodingConfig()
    assert config.alphas == DEFAULT_ALPHAS
    assert config.beam_size == 4


def test_ensemble_groups_duplicate_token_probabilities() -> None:
    original = torch.tensor([2.0, 1.9, 0.0])
    masked = torch.tensor([2.0, 0.0, 0.0])
    candidates = ensemble_candidates(
        original,
        masked,
        alphas=(0.0, 0.01, 1.0),
    )

    assert [candidate.token_id for candidate in candidates] == [0, 1]
    probability_at_zero = torch.softmax(original, dim=-1)[0].item()
    probability_at_point_zero_one = torch.softmax(
        original + 0.01 * (original - masked),
        dim=-1,
    )[0].item()
    assert candidates[0].probability == pytest.approx(
        (probability_at_zero + probability_at_point_zero_one) / 2
    )


def test_decoder_emits_exact_maximum_number_of_tokens() -> None:
    model = ConstantModel(token_id=1)
    decoder = IntentDecoder(
        model,
        DecodingConfig(
            alphas=(0.0,),
            beam_size=1,
            max_new_tokens=3,
        ),
    )

    generation = decoder.generate(prompt_views(), eos_token_ids=3)[0]

    assert generation.token_ids == (1, 1, 1)
    assert generation.steps == 3
    assert not generation.finished
    assert model.calls == 3


def test_decoder_stops_when_first_token_is_eos() -> None:
    model = ConstantModel(token_id=3)
    decoder = IntentDecoder(
        model,
        DecodingConfig(
            alphas=(0.0,),
            beam_size=1,
            max_new_tokens=3,
        ),
    )

    generation = decoder.generate(prompt_views(), eos_token_ids=3)[0]

    assert generation.token_ids == ()
    assert generation.steps == 1
    assert generation.finished
    assert model.calls == 1


def test_decoder_supports_transformers_dynamic_cache() -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).eval()
    decoder = IntentDecoder(
        model,
        DecodingConfig(
            alphas=(0.0,),
            beam_size=1,
            max_new_tokens=2,
        ),
    )

    generation = decoder.generate(prompt_views(), eos_token_ids=100)[0]

    assert len(generation.token_ids) == 2
    assert generation.steps == 2
