# \[Re] FairDICE: A Gap Between Theory And Practice

This repository contains code for running experiments and plotting graphs for our replication study "[\[Re\] FairDICE: A Gap Between Theory And Practice](https://arxiv.org/abs/2603.03454)", which aims to replicate the paper "[FairDICE: Fairness-Driven Offline Multi-Objective Reinforcement Learning](https://arxiv.org/abs/2506.08062v2)" by Kim et al.

As part of our replication effort, we discovered a critical bug in the original code and clarified certain parts of the original experiments, after which we were able to recreate some of the results from the original paper and verify the validity of the theory behind FairDICE. We also performed additional experiments using more complex enviornments and edge cases to test the generalisability of the algorithm.

The original code is available at: [ku-dmlab/FairDICE](https://github.com/ku-dmlab/FairDICE/tree/main) (note that we based our code on revision [bf610ab](https://github.com/ku-dmlab/FairDICE/tree/bf610ab4839177b9c8eb82f158cf3911b31a6f74))

To run experiments, please refer to the README.md files in the corresponding subdirectories. 

- [continuous-baselines](continuous-baselines) Code for running baselines in the continuous case
- [discrete](discrete) Contains code for reproducing and extending the results on discrete datasets
- [fairdice-d4morl-groupfair](fairdice-d4morl-groupfair) JAX code for FairDICE in continuous environments, minor modifications to run experiments in replication study, and code for running the GroupFair environment.
  - [graphs](fairdice-d4morl-groupfair/graphs) Code for plotting results and figures found in the report.
- [preference_weight_filtering](preference_weight_filtering) Code for performing the filtering operations for preference weights, for Appendix J.
- [pytorch-rewrite](pytorch-rewrite) Rewrite of FairDICE code using newer versions of libraries. Doesn't perfectly match results, not used in replication study. 
- [tests](tests) Code for the statistical tests on the beta hyperparameter in Appendix E.

# Citing

If you found our work useful, please cite the following (preprint):

```
@misc{adema2026refairdicegaptheory,
      title={[Re] FairDICE: A Gap Between Theory And Practice}, 
      author={Peter Adema and Karim Galliamov and Aleksey Evstratovskiy and Ross Geurts},
      year={2026},
      eprint={2603.03454},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.03454}, 
}
```
