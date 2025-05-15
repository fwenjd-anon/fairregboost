# FairRegBoost: An End-to-End Data Processing Framework for Fair and Scalable Regression

This repository contains the codes needed to reproduce the experiments of our submitted CIKM 2025 Paper

## General Information

- For the experiments we reuse work from other GitHubs (put in subfolders of algorithms)<br />
- For some approaches, some slight adaptations had to be made to integrate them in our framework, but the general approach was not altered.<br />
- The experiments were run under MacOS Sonoma 14.4, Python Version 3.9.6.

## HOWTO RUN

- You need to run `run_exps_reg.py`. The parameters to set for the experiments, like datasets, models to train, are in that file.<br />
- This will call the `main.py` function that runs the experiments, and subsequently calls `evaluation_reg.py` to evaluate the overall results.<br />
- An evaluation file is automatically generated for an experiment, which shows metrics for RMSE and W2, along additional metrics.
