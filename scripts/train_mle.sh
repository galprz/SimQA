#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate SimQA-code
echo "hello from $(python --version) in $(which python)"
echo "running train_mle.py experiments"

# Run the experiments
# python train_mle.py --mode v1 --train-steps 10000 --valid-steps 1000 --report-every 100 --save-every 1000
python train_mle.py --mode v2 --train-steps 10000 --valid-steps 1000 --report-every 100 --save-every 1000 --lookahead
