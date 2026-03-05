#!/usr/bin/env python3
import lzma
import multiprocessing
import os
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset

HERE = Path(__file__).resolve().parent

output_dir = HERE / "./compression_challenge_submission/"


def compress_tokens(tokens: np.ndarray) -> bytes:
    tokens = (
        tokens.astype(np.int16).reshape(-1, 128).T.ravel().tobytes()
    )  # transposing increases compression rate ;)
    return lzma.compress(tokens)


def compress_example(example):
    tokens = np.array(example["token.npy"])
    name = example["json"]["file_name"]  # or example['__key__']
    compressed = compress_tokens(tokens)
    compression_rate = (tokens.size * 10 / 8) / len(compressed)  # 10 bits per token
    with open(output_dir / name, "wb") as f:
        f.write(compressed)
    example["compression_rate"] = compression_rate
    return example


def compress_example_batched(batch):
    compression_rates = []

    for tokens, meta in zip(batch["token.npy"], batch["json"]):
        tokens = np.array(tokens)
        name = meta["file_name"]

        compressed = compress_tokens(tokens)
        compression_rate = (tokens.size * 10 / 8) / len(compressed)

        with open(output_dir / name, "wb") as f:
            f.write(compressed)

        compression_rates.append(compression_rate)

    batch["compression_rate"] = compression_rates
    return batch


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    num_proc = multiprocessing.cpu_count()

    # load split 0 and 1
    data_files = {"train": ["data-0000.tar.gz", "data-0001.tar.gz"]}
    print("Loading dataset...")
    ds = load_dataset("commaai/commavq", num_proc=num_proc, data_files=data_files)
    print("Dataset loaded.")
    print(f"Number of examples: {ds.num_rows}, compressing...")
    # compress
    ratios = ds.map(
        compress_example_batched,
        desc="compress_example",
        num_proc=num_proc,
        load_from_cache_file=False,
        batched=True,
        batch_size=32,
    )

    # make archive
    shutil.copy(HERE / "decompress.py", output_dir)
    shutil.make_archive(HERE / "compression_challenge_submission", "zip", output_dir)

    # print compression rate
    rate = (sum(ds.num_rows.values()) * 1200 * 128 * 10 / 8) / os.path.getsize(
        HERE / "compression_challenge_submission.zip"
    )
    print(f"Compression rate: {rate:.1f}")
