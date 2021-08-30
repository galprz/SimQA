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
To re-train the MLE models, run the train mle v1/v2 scripts:
```shell
python train_mle.py 
```
Arguments:
1. `--train-batch-size`: Batch size for training.
2. `--valid-batch-size`: Batch size for validation.
3. `--train-steps`: Number of training steps.
4. `--valid-steps`: Number of steps between evaluations on the validation set.
5. `--learning-rate`: Model's learning rate.
6. `--max-grad-norm`: Maximal gradients norm. Gradient above this threshold will be clipped.
7. `--report-every`: Number of steps between reports.
8. `--save-every`: Number of steps between saving models.

## The RL model
To train & evaluate the model using the reinforcement-learning algorithm and the Q semantic + syntactic reward,
download the pre-trained model from [here](https://drive.google.com/file/d/1z3wrQZe0V5HSXSiXyiIza36w-2zf6mmB/view?usp=sharing),
and place it in the root folder of the project. Then, run the following command:
```shell
python train_rl --data_version=v1
```