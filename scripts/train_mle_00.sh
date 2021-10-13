#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate SimQA-code
echo "hello from $(python --version) in $(which python)"
echo "running train_mle.py experiments"

# Run the experiments
#python train_mle.py --mode v1 --valid-steps 1000 --report-every 100 --name baseline
#python train_mle.py --mode v2 --valid-steps 1000 --report-every 100 --name baseline
python train_mle.py --mode v3 --valid-steps 1000 --report-every 100 --name baseline
