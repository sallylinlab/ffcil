# Training Script

### CIFAR100
stage 1

    python train.py -project base -dataset cifar100  -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 200 -schedule Cosine -gpu 0,1 -temperature 16 -mix 0.5
stage 2

    python train.py -project cec -dataset cifar100 -epochs_base 100 -episode_way 15 -episode_shot 1 -low_way 15 -low_shot 1 -lr_base 0.002 -lrg 0.0002 -step 20 -gamma 0.5 -gpu 0,1 -model_dir /params/cifar100_pretrain.pth

### Stanford Dogs
stage 1

    python train.py -project base -dataset stanford_dogs  -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 200 -schedule Cosine -gpu 0,1 -temperature 16 -mix 0.5
stage 2

    python train.py -project cec -dataset stanford_dogs -epochs_base 100 -episode_way 15 -episode_shot 1 -low_way 15 -low_shot 1 -lr_base 0.002 -lrg 0.0002 -step 20 -gamma 0.5 -gpu 0,1 -model_dir /params/stanford_dogs_pretrain.pth

### AUO
stage 1

    python train.py -project base -dataset auo -base_mode 'ft_cos' -new_mode 'avg_cos' -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 200 -schedule Cosine -gpu 0 -temperature 16 -mix 0.5
stage 1 DCL

    python train.py -project base -dataset auo -base_mode 'ft_cos' -new_mode 'avg_cos' -lr_base 0.001 -lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Cosine -gpu 0 -temperature 16 -mix 0.5 -model_dir params/resnet18_dcl_15c_cosine_AsoftLinear.pth -batch_size_base 64 -batch_size_new 512
stage 2

    python train.py -project cec -dataset auo -epochs_base 0 -epochs_res 100 -episode_way 5 -episode_shot 1 -low_way 5 -low_shot 1 -episode_query 10 -lr_base 0.002 -lrg 0.001 -step 20 -gpu 0,1 -model_dir params/auo_pretrain.pth





# Code and Files Explanation

## data/
### data/index_list
Stores the filenames of training data for different incremental stages
### dataloader/data_utils.py
Set up incremental scenarios for different datasets:  
base_class = Number of classes in the initial stage.  
num_classes = Number of classes after incremental stages. 
way = Number of classes added in each incremental stage.  
shot = Number of training samples per class.  
session = Total number of stages, including the initial stage.
### dataloader/[dataset name]/[data setname].py
Set the training dataset path.

## models/
### base/
Stage 1 training, also known as the joint training stage, performs standard CNN training to obtain a basic feature extraction model.
#### The training parameters for stage 1 (base) in train.py can be referenced from the script above. 
The mix parameter specifies the ratio of the mixup loss function, where 0<mix<1.

fscil_trainer.py is the main training code.

### cec/
Stage 2 training, also known as the pseudo incremental stage, fine-tunes the model from the previous stage. It continuously simulates new classes and trains them alongside old classes, utilizing the concept of meta-learning with multiple tasks.

#### The training parameters for stage 2 (CEC) in train.py can be referenced from the script above. 
-epochs_base :The number of epochs for the original method.  
-epochs_res: The number of epochs for the adjusted tasks in the new method. Typically, base is set to 0, and res is set to the desired number of training epochs. 
-episode_way: The number of classes to select for each task.
-episode_shot: The number of samples to select per class for each task.
-low_way: The number of classes to simulate for each task, which must be less than or equal to -episode_way.
-low_shot: The number of samples to simulate per class for the simulated classes in each task.
-episode_query: The number of samples required per class as query (test) for each task.
-pseudo_mode: The method for simulating new classes, with r representing rotation and c representing color jitter. The default is rotate.

fscil_trainer.py is the main training code.

# Environment
    conda env create -f environment.yml

An environment named auofscil will be automatically created.