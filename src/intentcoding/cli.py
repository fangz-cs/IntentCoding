"""Command-line interface for IntentCoding inference."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import json
from pathlib import Path
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intentcoding.decoding import (
    DEFAULT_ALPHAS,
    DecodingConfig,
    IntentDecoder,
)
from intentcoding.masking import MASK_MARKER, encode_prompt_views


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Intent-amplified decoding for code generation."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or path")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--id-key", default="task_id")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--masked-prompt-key", default="masked_prompt")
    parser.add_argument("--mask-marker", default=MASK_MARKER)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
    )
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or a concrete device such as 'cuda:1'",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--resume", action="store_true")
    output_mode.add_argument("--overwrite", action="store_true")
    return parser


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: each row must be a JSON object"
                )
            yield line_number, value


def _require_text(
    record: dict[str, Any],
    key: str,
    path: Path,
    line_number: int,
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{path}:{line_number}: {key!r} must be a non-empty string"
        )
    return value


def _completed_ids(path: Path, id_key: str) -> set[str]:
    completed = set()
    if not path.exists():
        return completed
    for line_number, record in _iter_jsonl(path):
        if id_key not in record:
            raise ValueError(
                f"{path}:{line_number}: missing output ID key {id_key!r}"
            )
        completed.add(str(record[id_key]))
    return completed


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _eos_token_ids(model: Any, tokenizer: Any) -> int | list[int]:
    value = getattr(model.generation_config, "eos_token_id", None)
    if value is None:
        value = tokenizer.eos_token_id
    if value is None:
        raise ValueError("model and tokenizer do not define an EOS token")
    if isinstance(value, int):
        return value
    return [int(token_id) for token_id in value]


def _load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    model.to(_resolve_device(args.device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    return model, tokenizer


def run(args: argparse.Namespace) -> None:
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output paths must differ")
    if not args.input.is_file():
        raise FileNotFoundError(f"input file does not exist: {args.input}")
    if args.output.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"output already exists: {args.output}; use --resume or --overwrite"
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = DecodingConfig(
        alphas=tuple(args.alphas),
        beam_size=args.beam_size,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
    )
    model, tokenizer = _load_model_and_tokenizer(args)
    decoder = IntentDecoder(model, config)
    eos_token_ids = _eos_token_ids(model, tokenizer)

    completed = _completed_ids(args.output, args.id_key) if args.resume else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output.exists() else "w"
    processed = 0

    with args.output.open(mode, encoding="utf-8") as output:
        for line_number, record in _iter_jsonl(args.input):
            if args.id_key not in record:
                raise ValueError(
                    f"{args.input}:{line_number}: "
                    f"missing ID key {args.id_key!r}"
                )
            record_id = record[args.id_key]
            if str(record_id) in completed:
                continue

            prompt = _require_text(
                record,
                args.prompt_key,
                args.input,
                line_number,
            )
            masked_prompt = _require_text(
                record,
                args.masked_prompt_key,
                args.input,
                line_number,
            )
            views = encode_prompt_views(
                tokenizer,
                prompt,
                masked_prompt,
                marker=args.mask_marker,
            )
            generations = decoder.generate(views, eos_token_ids)
            result = {
                args.id_key: record_id,
                "completions": [
                    {
                        "text": tokenizer.decode(
                            generation.token_ids,
                            skip_special_tokens=True,
                        ),
                        "score": generation.score,
                        "mean_log_probability": (
                            generation.mean_log_probability
                        ),
                        "finished": generation.finished,
                        "steps": generation.steps,
                    }
                    for generation in generations
                ],
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            completed.add(str(record_id))
            processed += 1
            print(f"processed {processed} record(s)", file=sys.stderr)


def main() -> None:
    run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
