# SimQA
This is the code for the SimQA paper.

## Installations Guide
1. Install an environment manager. Recommeneded: [Miniconda3](https://docs.conda.io/en/latest/miniconda.html).
   Here is a [Getting Started](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda) guide.
2. Clone the repo:
   ```sh
   git clone https://github.com/nitaifingerhut/SimQA.git
   cd SimQA
   ```
3. Create a new environment from environment.yml (you can change the environment name in the file)
   ```sh
   conda env update -f environment.yml
   conda activate SimQA-code
   ```
4. On **MacOS**, run: 
    ```shell
    conda install nomkl
    ```
   
## Dataset
This paper uses a unique dataset that was labeled by chemists, and it's located in `data` directory.

## The MLE model
To re-train the MLE models, run:
```shell
python train_mle.py 
```
Arguments:
1. `--mode` Version of the data (v1/v2).
2. `--train-batch-size` Batch size for training.
3. `--valid-batch-size` Batch size for validation.
4. `--train-steps` Number of training steps.
5. `--valid-steps` Number of steps between evaluations on the validation set.
6. `--learning-rate` Model's learning rate.
7. `--max-grad-norm` Maximal gradients norm. Gradient above this threshold will be clipped.
8. `--report-every` Number of steps between reports.
9. `--save-every` Number of steps between saving models.

Saved models will be under `checkpoints/<mode>_step_<i * save-every>`, for `0 <= i < <train-steps> // <save-every>`.

## The RL model
To train and evaluate the model using the reinforcement-learning algorithm and the Q semantic + syntactic reward, run:
```shell
python train_rl.py --data_version=v1
```