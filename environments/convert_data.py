"""Utility for stacking D4RL dataset into contiguous arrays of single precision"""

import argparse
import gc
import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm.auto as tqdm


def numpy_to_pyarrow(arr: npt.NDArray) -> pa.Table:
    if len(arr.shape) == 1:
        arr = arr.reshape(-1, 1)
    assert len(arr.shape) == 2
    return pa.table({str(i): arr[:, i] for i in range(arr.shape[1])})


def stack_rollouts(path: str | Path) -> dict[str, pa.Table]:
    with Path(path).open("rb") as f:
        batched = pickle.load(f)
    unstacked = {key: [] for key in batched[0]}
    unstacked["init_idxs"] = []
    obs_idx = 0
    for traj in batched:
        for key, val in traj.items():
            unstacked[key].append(val)
        traj_inits = np.full_like(traj["terminals"], obs_idx, dtype=np.uint32)
        unstacked["init_idxs"].append(traj_inits)
        obs_idx += len(traj["terminals"])
    unstacked = list(reversed(unstacked.items()))
    stacked = {}

    dtypes = {
        "actions": np.float32,
        "next_observations": np.float32,
        "observations": np.float32,
        "preference": np.float32,
        "raw_rewards": np.float32,
        "terminals": np.bool,
        "init_idxs": np.uint32,
    }
    while unstacked:
        gc.collect()
        key, val = unstacked.pop()
        stacked[key] = numpy_to_pyarrow(np.concat(val, dtype=dtypes.pop(key)))

    assert not dtypes, f"Didn't see {', '.join(dtypes.keys())}?"

    return stacked


def preprocess_d4rl(input_root: str | Path, output_root: str | Path):
    output_root = Path(output_root)
    for path in tqdm.tqdm(list(Path(input_root).glob("*/*.pkl"))):
        out = output_root / path.parent.name / path.with_suffix("").name
        if out.exists():
            print("Skipping", out, "exists")
            continue

        stacked = stack_rollouts(path)
        out.mkdir(exist_ok=True, parents=True)
        for name, table in stacked.items():
            pq.write_table(table, f"{out}/{name}.pq", compression=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Original dataset", required=True)
    parser.add_argument("--output", type=str, help="Output directory", required=True)
    args = parser.parse_args()
    preprocess_d4rl(args.input, args.output)
