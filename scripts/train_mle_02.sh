#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate SimQA-code
echo "hello from $(python --version) in $(which python)"
echo "running train_mle.py experiments"

# Run the experiments
python train_mle.py --version v3 --valid-steps 1000 --report-every 100 --lookahead --lookahead-steps 05 --lookahead-alpha 0.10 --name lookahead_ls_05_la_0.10
python train_mle.py --version v3 --valid-steps 1000 --report-every 100 --lookahead --lookahead-steps 10 --lookahead-alpha 0.10 --name lookahead_ls_10_la_0.10
python train_mle.py --version v3 --valid-steps 1000 --report-every 100 --lookahead --lookahead-steps 15 --lookahead-alpha 0.10 --name lookahead_ls_15_la_0.10
