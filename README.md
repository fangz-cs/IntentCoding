# IntentCoding

[![CI](https://github.com/fangz-cs/IntentCoding/actions/workflows/ci.yml/badge.svg)](https://github.com/fangz-cs/IntentCoding/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **IntentCoding: Amplifying User Intent in Code Generation** (ACL 2026).

IntentCoding is a training-free decoding method that strengthens the influence
of a natural-language intent on code generation. At every decoding step, it
computes logits for two views of the same prefix:

1. the original prompt; and
2. the same prompt with attention to the intent span disabled.

If `o` and `o_masked` are the two logit vectors, the intent signal is

```text
delta = o - o_masked
```

For each `alpha` in `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`, IntentCoding selects the
top token from `o + alpha * delta`. Duplicate token IDs are grouped and scored
by the mean of the softmax probabilities under the strengths that selected
them. The unique candidates expand the current hypotheses, which are pruned by
cumulative log-probability. Final hypotheses that reached EOS are preferred and
ranked by mean log-probability.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development:

```bash
pip install -e ".[test]"
pytest
```

The implementation targets decoder-only Hugging Face models and requires
access to their output logits. The ACL 2026 experiments use:

- `codellama/CodeLlama-7b-hf`
- `deepseek-ai/deepseek-coder-6.7b-base`
- `Qwen/Qwen2.5-Coder-7B`



## Input format

The command-line interface reads JSON Lines. Each record must contain an ID,
an original prompt, and a masked prompt. Replace exactly one contiguous intent
span with `<mask_ins>`; all text before and after that span must remain
unchanged.

```json
{"task_id":"example/0","prompt":"Write a function that returns exactly two positive integers.","masked_prompt":"Write a function that <mask_ins>."}
```

The masked text itself is never passed to the model. It is tokenized only to
locate the corresponding positions in the original prompt, then those
positions are set to zero in the second attention mask.

Run decoding with:

```bash
intentcoding \
  --model Qwen/Qwen2.5-Coder-7B \
  --input data/input.jsonl \
  --output outputs/predictions.jsonl \
  --max-new-tokens 256
```

Use `--prompt-key`, `--masked-prompt-key`, and `--id-key` when a benchmark uses
different field names. Existing output is never overwritten unless
`--overwrite` is given; `--resume` appends only IDs not already present.

Each output record contains the source ID and ranked completions:

```json
{"task_id":"example/0","completions":[{"text":"...","score":-1.23,"mean_log_probability":-0.12,"finished":true,"steps":10}]}
```

Prompts are not copied to the output or printed by the CLI.

## Build CodeConstraints

The release includes a deterministic refactor of the original dataset builder.
It preserves the paper's prompt templates and JSON schema while removing
hard-coded paths and implicit global random state.

```bash
intentcoding-build-codeconstraints \
  --output-dir data/codeconstraints \
  --seed 42
```

The default command creates the five paper splits:

| File | Records |
| --- | ---: |
| `level2_datatype_without_sys.jsonl` | 100 |
| `level2_len_without_sys.jsonl` | 100 |
| `level2_size_without_sys.jsonl` | 100 |
| `level3_without_sys.jsonl` | 100 |
| `level4_mask_all_without_sys.jsonl` | 100 |

Level 4 records contain `prompt_mask` for the full intent as well as
`prompt_mask_size`, `prompt_mask_len`, and `prompt_mask_sizeandlen` for the
fine-grained masking experiments. Generated JSONL files remain ignored by Git.

To decode the generated Level 4 split:

```bash
intentcoding \
  --model Qwen/Qwen2.5-Coder-7B \
  --input data/codeconstraints/level4_mask_all_without_sys.jsonl \
  --output outputs/codeconstraints-qwen.jsonl \
  --masked-prompt-key prompt_mask \
  --max-new-tokens 256
```

## Paper settings


| Setting                       | Value                                          |
| ----------------------------- | ---------------------------------------------- |
| Amplification strengths       | `0.0 0.2 0.4 0.6 0.8 1.0`                      |
| Token ensemble                | Mean probability for duplicate top-1 token IDs |
| Beam size                     | `4`                                            |
| Beam pruning                  | Cumulative log-probability                     |
| HumanEval / IFEvalCode length | `512`                                          |
| LiveCodeBench length          | `1024`                                         |
| CodeConstraints length        | `256`                                          |


The repository intentionally contains no benchmark data, model checkpoints,
generated samples, logs, or experimental results. Obtain HumanEval,
LiveCodeBench, IFEvalCode, and CodeConstraints from their respective releases,
then construct masked prompts following Appendix F of the paper.

## Repository layout

```text
src/intentcoding/
  codeconstraints.py  Deterministic CodeConstraints construction
  cli.py       Generic JSONL inference command
  decoding.py  Multi-strength ensemble and beam search
  masking.py   Intent-span attention masking
scripts/
  build_codeconstraints.py  Standalone builder entry point
tests/         Deterministic unit tests with synthetic inputs
```



## Citation

```bibtex
@inproceedings{fang-etal-2026-intentcoding,
  title = {{IntentCoding}: Amplifying User Intent in Code Generation},
  author = {Fang, Zheng and Dong, Yihong and Mou, Lili and
            Jin, Dongming and Jin, Zhi and Li, Ge},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year = {2026},
  pages = {33246--33261},
  doi = {10.18653/v1/2026.findings-acl.1662},
  url = {https://aclanthology.org/2026.findings-acl.1662/}
}
```

- Paper: [https://aclanthology.org/2026.findings-acl.1662/](https://aclanthology.org/2026.findings-acl.1662/)
- DOI: [https://doi.org/10.18653/v1/2026.findings-acl.1662](https://doi.org/10.18653/v1/2026.findings-acl.1662)
