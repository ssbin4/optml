# OptML Final Project

This is the final project code for the CSED490Y Optimization for Machine Learning of Team 01.
The project aims to analyze the effect of learning rate scheduling on transfer learning.
For the experiment, we used Resnet18 model pretrained with Imagenet, and chose Stanford CARS196 as the target dataset.
Much of the code is based on [https://github.com/sigopt/stanford-car-classification](https://github.com/sigopt/stanford-car-classification).

### Prepare data

```
mkdir CARS196
cd CARS196
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
tar -xvzf cars_train.tgz
tar -xvzf car_devkit.tgz
```

After downloading and unzipping the data, folder heirarchy should be as below.
```
+-- CARS196
    +-- cars_train
        +-- 00001.jpg
        +-- 00002.jpg
        ...
        +-- 08144.jpg
    +-- devkit
        +-- cars_meta.mat
        +-- cars_train_annos.mat
```

### Prepare virtual environment

First, change the prefix in environment.yaml file to your path.
For example, prefix: /home/<your_id>/.conda/envs/cbm

Then, to install the libraries,
```
conda env create --file environment.yaml
```

We used python version 3.6.13 for the experiment.


To activate the environment,
```
source activate optml
```

### WandB (Weights and Biases)

To visualsize training loss or accuracy, we used wandb.

To install wandb and login, type as follows.
```
pip3 install wandb
wandb login
```

### Training

```
python run_resnet_training_cli.py 
--path_images ./CARS196/cars_train 
--path_data ./CARS196/devkit/cars_train_annos.mat 
--path_labels ./CARS196/devkit/cars_meta.mat 
--path_model_checkpoint ./checkpoint/ 
--checkpoint_frequency <frequency to generate PyTorch checkpoint files>
--model ResNet18 
--epochs <number of epochs to train model>
--validation_frequency <frequency to run validation during training>
--number_of_classes 196 
--data_subset 1.0 
--batch_size <batch size>
--weight_decay <weight decay(L2 regularziation loss)>
--lr_scheduler <learning scheduler to use. choose in {const, step, exp, cos, reduce}>
--annealing_factor <new_learning rate = annealing_factor * current_learning_rate in 'step' and 'reduce'>
--momentum 0.9 <momentum in SGD>
--init_lr 0.0001 <initial learning rate value>
--scheduler_rate 10 <interval to apply learning rate scheduling in 'step' and 'reduce'>
--no-nesterov
{--freeze_weights | --no-freeze_weights} <whether to freeze the pretrained network or not>
{--diff_lr | --same-lr} <whether to use different inital learning rate for the pretrained network and final FC layer or not>
--gamma 0.990 <gamma value used in 'exp'>
```

### Experimented learning rate scheudlers

1. Constant learning rate ('const'): Learning rate is always same to the initial value.
2. Step learning rate decay ('step'): After every scheduler_rate epochs, new_learning_rate = <annealing_factor> * current_learning_rate. We used torch.optim.lr_scheduler.stepLR.
3. Exponential learning rate decay ('exp'): For each epoch, new_learning_rate = <gamma> * current_learning_rate. We used torch.optim.lr_scheduler.exponentialLR.
4. Cosine annealing warm restarts ('cos'): This scheduler was proposed in [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983). We used torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.
5. Reduce LR on plateau ('reduce'): If training loss does not improve for <scheduler_rate> epochs, new_learning rate = <annealing_factor>*current_learning_rate. We used torch.optim.lr_scheduler.reduceLROnPlateau.