"""Utility for stacking D4RL dataset into contiguous arrays"""

import argparse
import gc
import pickle
from pathlib import Path

import numpy as np
import tqdm.auto as tqdm


def stack_rollouts(path: str | Path):
    with Path(path).open("rb") as f:
        batched = pickle.load(f)
    unstacked = {key: [] for key in batched[0]}
    for traj in batched:
        for key, val in traj.items():
            unstacked[key].append(val)
    unstacked = list(reversed(unstacked.items()))
    stacked = {}
    while unstacked:
        gc.collect()
        key, val = unstacked.pop()
        stacked[key] = np.concat(val)

    return stacked


def preprocess_d4rl(input_root: str | Path, output_root: str | Path):
    output_root = Path(output_root)
    for path in tqdm.tqdm(list(Path(input_root).glob("*/*.pkl")), desc="Preprocessing"):
        out = output_root / path.parent.name / path.with_suffix(".npz").name
        out.parent.mkdir(exist_ok=True, parents=True)
        if out.exists():
            print("Skipping", out, "exists")
            continue

        stacked = stack_rollouts(path)
        np.savez(out, **stacked)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Original dataset", required=True)
    parser.add_argument("--output", type=str, help="Output directory", required=True)
    args = parser.parse_args()
    preprocess_d4rl(args.input, args.output)
