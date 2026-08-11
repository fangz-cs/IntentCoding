import pytest
import torch

from intentcoding.masking import MASK_MARKER, encode_prompt_views


class CharacterTokenizer:
    def __init__(self) -> None:
        self.vocab = {MASK_MARKER: 1}

    def get_vocab(self):
        return dict(self.vocab)

    def add_special_tokens(self, mapping):
        for token in mapping["additional_special_tokens"]:
            self.vocab.setdefault(token, len(self.vocab) + 1)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def __call__(self, text, return_tensors):
        assert return_tensors == "pt"
        token_ids = [0]
        index = 0
        while index < len(text):
            if text.startswith(MASK_MARKER, index):
                token_ids.append(self.vocab[MASK_MARKER])
                index += len(MASK_MARKER)
            else:
                token_ids.append(ord(text[index]) + 10)
                index += 1
        values = torch.tensor([token_ids])
        return {
            "input_ids": values,
            "attention_mask": torch.ones_like(values),
        }


class OffsetTokenizer:
    is_fast = True

    def __call__(self, text, return_tensors, return_offsets_mapping):
        assert return_tensors == "pt"
        assert return_offsets_mapping
        values = torch.tensor([[0, 1, 2, 3]])
        return {
            "input_ids": values,
            "attention_mask": torch.ones_like(values),
            "offset_mapping": torch.tensor(
                [[[0, 0], [0, 6], [6, 12], [12, len(text)]]]
            ),
        }


def test_encode_prompt_views_masks_only_replaced_span() -> None:
    views = encode_prompt_views(
        CharacterTokenizer(),
        "prefixINTENTsuffix",
        f"prefix{MASK_MARKER}suffix",
    )

    assert views.input_ids.shape == views.attention_mask.shape
    assert views.masked_attention_mask[0, 1:7].tolist() == [1] * 6
    assert views.masked_attention_mask[0, 7:13].tolist() == [0] * 6
    assert views.masked_attention_mask[0, 13:].tolist() == [1] * 6


def test_encode_prompt_views_rejects_changed_context() -> None:
    with pytest.raises(ValueError, match="before the mask marker"):
        encode_prompt_views(
            CharacterTokenizer(),
            "prefixINTENTsuffix",
            f"changed{MASK_MARKER}suffix",
        )


def test_fast_tokenizer_uses_character_offsets() -> None:
    views = encode_prompt_views(
        OffsetTokenizer(),
        "prefixINTENTsuffix",
        f"prefix{MASK_MARKER}suffix",
    )

    assert views.masked_attention_mask.tolist() == [[1, 1, 0, 1]]
