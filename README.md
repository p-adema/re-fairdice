# Re: FairDICE

This is a replication study of the [FairDice](https://openreview.net/forum?id=2jQJ7aNdT1) 
paper on balancing multiple objectives in offline reinforcement learning.

## Setup:
This project uses the [uv](https://docs.astral.sh/uv/#installation) package manager,
which automatically installs all packages when used. 

The dataset is slightly different from the original D4RL dataset 
(stored in single precision for storage efficiency, and in Apache Parquet to minimise
the use of pickle files); the original D4RL dataset can be converted by running:

```shell
uv run environments/convert_data.py --input <original data path> --output data
```

Alternatively, it can be downloaded from HuggingFace (not yet),

```shell
uvx --with huggingface_hub hf <dataset link to come here>
```

## Experiments
Experiments for replicating Fig. 8 from Appendix I of FairDICE can be run using the 
`run_all.sh` script, by running

```shell
./run_all.sh
```

Individual experiments can be run using `main.py`, e.g.

```shell
uv run main.py --env_name MO-Ant-v2 --beta 0.1
```