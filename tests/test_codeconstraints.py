import json
from pathlib import Path
import random

import pytest

from intentcoding.codeconstraints import (
    BuildConfig,
    SPLIT_FILENAMES,
    build_splits,
    generate_level2,
    write_splits,
)
from intentcoding.masking import MASK_MARKER


def assert_valid_mask(prompt: str, masked_prompt: str) -> None:
    assert masked_prompt.count(MASK_MARKER) == 1
    prefix, suffix = masked_prompt.split(MASK_MARKER)
    assert prompt.startswith(prefix)
    assert prompt.endswith(suffix)
    assert len(prompt) > len(prefix) + len(suffix)


def test_level2_datatype_preserves_original_template() -> None:
    record = generate_level2(
        count=1,
        subset="datatype",
        rng=random.Random(0),
    )[0]

    assert record == {
        "task_id": "level_2_task_0",
        "prompt": (
            "\ndef func():\n"
            "    '''Return a list of floats.\n"
            "    '''\n"
        ),
        "prompt_mask": (
            "\ndef func():\n"
            "    '''<mask_ins>\n"
            "    '''\n"
        ),
        "constraints": {"dataType": "float"},
    }


def test_build_splits_matches_paper_structure() -> None:
    config = BuildConfig(
        seed=7,
        level2_per_subset=3,
        level3_size=4,
        level4_size=5,
    )
    splits = build_splits(config)

    assert list(splits) == list(SPLIT_FILENAMES)
    assert [len(rows) for rows in splits.values()] == [3, 3, 3, 4, 5]
    assert all(
        len(record["constraints"]) == 3
        for record in splits["level3"]
    )
    assert all(
        len(record["constraints"]) == 4
        for record in splits["level4"]
    )

    for rows in splits.values():
        for record in rows:
            assert_valid_mask(record["prompt"], record["prompt_mask"])

    for record in splits["level4"]:
        assert_valid_mask(record["prompt"], record["prompt_mask_size"])
        assert_valid_mask(record["prompt"], record["prompt_mask_len"])
        assert_valid_mask(
            record["prompt"],
            record["prompt_mask_sizeandlen"],
        )


def test_build_is_deterministic_and_split_stable() -> None:
    baseline = build_splits(BuildConfig(seed=19, level3_size=2))
    repeated = build_splits(BuildConfig(seed=19, level3_size=2))
    resized = build_splits(BuildConfig(seed=19, level3_size=8))

    assert baseline == repeated
    assert baseline["level4"] == resized["level4"]
    assert baseline != build_splits(BuildConfig(seed=20, level3_size=2))


def test_write_splits_refuses_implicit_overwrite(tmp_path: Path) -> None:
    splits = build_splits(
        BuildConfig(
            seed=1,
            level2_per_subset=1,
            level3_size=1,
            level4_size=1,
        )
    )
    targets = write_splits(tmp_path, splits)

    for split, target in targets.items():
        rows = [
            json.loads(line)
            for line in target.read_text(encoding="utf-8").splitlines()
        ]
        assert rows == splits[split]

    with pytest.raises(FileExistsError):
        write_splits(tmp_path, splits)

    assert write_splits(tmp_path, splits, overwrite=True) == targets

