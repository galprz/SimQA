#!/bin/bash

sbatch -c 2 --gres=gpu:1 -o train_mle_00.out -J mle00 train_mle_00.sh
sbatch -c 2 --gres=gpu:1 -o train_mle_01.out -J mle01 train_mle_01.sh
sbatch -c 2 --gres=gpu:1 -o train_mle_02.out -J mle02 train_mle_02.sh
sbatch -c 2 --gres=gpu:1 -o train_mle_03.out -J mle03 train_mle_03.sh
sbatch -c 2 --gres=gpu:1 -o train_mle_04.out -J mle04 train_mle_04.sh
