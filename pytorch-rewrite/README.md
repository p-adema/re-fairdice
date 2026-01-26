# Re: FairDICE in PyTorch

This is an attempt at rewriting the [FairDice](https://openreview.net/forum?id=2jQJ7aNdT1) paper on balancing multiple 
objectives in offline reinforcement learning into PyTorch. The main structure for
replicating results on the D4RL dataset is complete, but in some environments
(primarily Walker2d) the results are slightly worse than the original. As such, it
can only serve as an informational supplement to the main replication study, not to
generate results or evaluate the FairDICE method.

### This code is provided for completeness, but is not how results were gathered for our replication study.

## Setup:
This project uses the [uv](https://docs.astral.sh/uv/#installation) package manager,
which automatically installs all packages when used. 

The dataset is slightly different from the original D4RL dataset 
(stored in single precision for storage efficiency, and in Apache Parquet to minimise
the use of pickle files); the original D4RL dataset can be converted by running:

```shell
uv run environments/convert_data.py --input <original data path> --output data
```

## Experiments
Experiments for replicating Fig. 8 from Appendix I of FairDICE can be run using the 
`run_all.sh` script, by running

```shell
./run_all.sh
```

Individual experiments can be run using `main.py`, e.g.

```shell
uv run main.py --env_name MO-Hopper-v2 --beta 0.1
```

# Results
After running the experiments, we obtained results for the fixed-broadcasting version
which are globally similar to the Jax-based FairDICE implementation, but not identical:

![Result graph](pt-rewrite-results.png)