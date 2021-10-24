#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate SimQA-code
echo "hello from $(python --version) in $(which python)"
echo "running train_mle.py experiments"

# Run the experiments
#python train_mle.py --version v2 --valid-steps 1000 --report-every 100 --train-batch-size 24 --name baseline
python train_rl.py --version v2 --model-path checkpoints/MLE/v2/baseline_step_10000.pt --train-steps 1000 --valid-steps 100 --report-every 100 --save-every 100 --name baseline