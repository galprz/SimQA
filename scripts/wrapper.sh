#!/bin/bash

sbatch -c 2 --gres=gpu:1 -o train.out -J SimQA train.sh
