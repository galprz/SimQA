#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate SimQA-code
echo "hello from $(python --version) in $(which python)"
echo "running train_mle.py experiments"

# Run the experiments
python train_rl.py --mode v1 --train-steps 1000 --valid-steps 100 --report-every 100 --save-every 100
