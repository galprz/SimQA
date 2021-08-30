# SimQA
This is the code for the SimQA paper.

## Dataset
This paper uses a unique dataset that was labeled by chemists. 
You need to download and extract the zip file, and place it in the root folder of the project. 
Download the zip from [here](https://drive.google.com/drive/folders/16XJkKMJj-cPqw_msczG1VZ7fOPOwgKnh?usp=sharing).

## The MLE model
To re-train the MLE models, run the train mle v1/v2 scripts.\
To train & evaluate the model using the reinforcement-learning algorithm and the Q semantic + syntactic reward,
download the pre-trained model from [here](https://drive.google.com/file/d/1z3wrQZe0V5HSXSiXyiIza36w-2zf6mmB/view?usp=sharing),
and place it in the root folder of the project. Then, run the following command:
```shell
python train_rl --data_version=v1
```