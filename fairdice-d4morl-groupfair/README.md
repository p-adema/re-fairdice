# Replication study of "FairDICE: Fairness-Driven Offline Multi-Objective Reinforcement Learning"
This directory contains the code to replicate results for FairDICE on
D4MORL tasks, as well as on the GroupFair task (note: in code, GroupFair 
is usually referred to as GroupPolicy instead).

INTERFACE CHANGES: in `main.py`, an additional argument `--loss_kind` is added, with values
`wrong-broadcast`, `behaviour-cloning` and `fixed-fairdice`, and a `--discrete` argument
is added to allow for running experiments using discrete policies 
(use `--env_name GroupPolicy-v1` to train GroupFair). `run_all.sh` is also 
changed to generate all results necessary to replicate our graphs.
Furthermore, a `post-eval.py` scripts is provided to re-run evaluations with more rollouts
using model checkpoints.

## Setup
  Use the offered Dockerfile for the setup and create conda environment using yml file.
  ```
  cd FairDICE
  conda env create -f environment.yml
  conda activate fairdice
  ```

## Data Download
This repository uses the D4MORL dataset, a benchmark suite designed for offline multi-objective reinforcement learning (MORL). The dataset was introduced in the following paper:

Zhu, Baiting, Meihua Dang, and Aditya Grover.
Scaling Pareto-Efficient Decision Making via Offline Multi-Objective RL.
The Eleventh International Conference on Learning Representations (ICLR), 2023.

D4MORL provides diverse multi-objective versions of standard MuJoCo locomotion tasks (e.g., Hopper, Walker2d, HalfCheetah), enabling the evaluation of Pareto-efficient and fairness-aware policies under offline constraints.

To download the data, run:
```
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1wfd6BwAu-hNLC9uvsI1WPEOmPpLQVT9k?usp=sharing --output data
```

## Training
If you want to run all experiments
```
./run_all.sh
```
or if you want to run a single experiment
```
CUDA_VISIBLE_DEVICES=0 python main.py --learner FairDICE --divergence SOFT_CHI --env_name MO-Hopper-v2 --quality expert --beta 0.1 --preference_dist uniform --eval_episodes 10 --batch_size 256 --hidden_dim 768 --num_layers 3 --total_train_steps 100000 --log_interval 1000 --normalize_reward  True
```

`run_all.sh` will run all experiments needed to reproduce FairDICE baselines, and at the
end also gathers results for GroupPolicy. Results are placed into subdirectories of
the `structured-results` directory; to draw graphs, the BASE_DIR in boxplots.ipynb
should be updated to point to your `structured-results` directory.

## License
The original FairDICE code, and also this replication study, is licensed under the MIT License.
