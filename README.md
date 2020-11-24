# SimQA
This is the code for the SimQA paper
# Dataset
This paper use a unique dataset that was labeled by chemists.
You need to extract the zip file to the root folder to run the scripts
# The MLE model
To run re-train the mle models run the train mle v1/v2 scripts
To train & evaluate the model using the  reinforcement-learning algorithm and the Q semantic + syntactic reward
download the pre-trained model from here: 
https://drive.google.com/file/d/1z3wrQZe0V5HSXSiXyiIza36w-2zf6mmB/view?usp=sharing
and put it in the root folder of the project.
Then run the following command:
```
python train_rl --data_version=v1
```