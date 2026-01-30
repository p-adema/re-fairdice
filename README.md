# \[Re] FairDICE: A Gap Between Theory And Practice

This repository contains code for running experiments and plotting graphs for our replication study "\[Re] FairDICE: A Gap Between Theory And Practice", which aims to replicate the paper "FairDICE: Fairness-Driven Offline Multi-Objective Reinforcement Learning" by Kim et al.

To run experiments, please refer to the README.md files in the corresponding subdirectories. 

- [continuous](continuous) Code for running baselines in the continuous case
- [discrete](discrete) Contains code for reproducing and extending the results on discrete datasets
- [fairdice-d4morl-groupfair](fairdice-d4morl-groupfair) JAX code for FairDICE in continuous environments, minor modifications to run experiments in replication study, and code for running the GroupFair environment.
- [graphs](graphs) Code for plotting results and figures found in the report.
- [preference_weight_filtering](preference_weight_filtering) Code for performing the filtering operations for preference weights, for Appendix J.
- [pytorch-rewrite](pytorch-rewrite) Rewrite of outdated FairDICE code using newer versions of libraries. Doesn't perfectly match results, not used in replication study. 
- [tests](tests) Code for the statistical tests on the beta hyperparameter in Appendix E.
