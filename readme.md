# FFCIL
Code release for FFCIL: Fine-grained Few-shot Class Incremental Learning

# Environment
A conda environment named auofscil can be created with the following command:

    conda env create -f environment.yml

# Dataset
Public datasets supporting the experiments can be downloaded at the following links:
* CUB_200_2011 [https://www.vision.caltech.edu/datasets/cub_200_2011/]
* StanforDogs [https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset]

# Training Script

### CUB200
* __stage 1__

        python train.py -project base -dataset cub200  -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 200 -schedule Cosine -gpu 0,1 -temperature 16 -mix 0.5
  
* __stage 2__

        python train.py -project cec -dataset cub200 -epochs_base 100 -episode_way 15 -episode_shot 1 -low_way 15 -low_shot 1 -lr_base 0.002 -lrg 0.0002 -step 20 -gamma 0.5 -gpu 0,1 -model_dir /params/cub200_pretrain.pth

### Stanford Dogs
* __stage 1__

        python train.py -project base -dataset stanford_dogs  -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 200 -schedule Cosine -gpu 0,1 -temperature 16 -mix 0.5
* __stage 2__

        python train.py -project cec -dataset stanford_dogs -epochs_base 100 -episode_way 15 -episode_shot 1 -low_way 15 -low_shot 1 -lr_base 0.002 -lrg 0.0002 -step 20 -gamma 0.5 -gpu 0,1 -model_dir /params/stanford_dogs_pretrain.pth

### AUO
* __stage 1__

        python train.py -project base -dataset auo -base_mode 'ft_cos' -new_mode 'avg_cos' -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 200 -schedule Cosine -gpu 0 -temperature 16 -mix 0.5

* __stage 1 DCL__

        python train.py -project base -dataset auo -base_mode 'ft_cos' -new_mode 'avg_cos' -lr_base 0.001 -lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Cosine -gpu 0 -temperature 16 -mix 0.5 -model_dir params/resnet18_dcl_15c_cosine_AsoftLinear.pth -batch_size_base 64 -batch_size_new 512

* __stage 2__

        python train.py -project cec -dataset auo -epochs_base 0 -epochs_res 100 -episode_way 5 -episode_shot 1 -low_way 5 -low_shot 1 -episode_query 10 -lr_base 0.002 -lrg 0.001 -step 20 -gpu 0,1 -model_dir params/auo_pretrain.pth


# Code and Files Explanation
## data/
### data/index_list
This folder contains filenames of training data for various incremental stages across different datasets.

### dataloader/data_utils.py
This file contains incremental scenario setups for different datasets with following parameters:
* __base_class__: number of classes in the initial stage.  
* __num_classes__: number of classes after incremental stages. 
* __way__: number of classes added in each incremental stage.  
* __shot__: number of training samples per class.  
* __session__: notal number of stages, including the initial stage.

### dataloader/[dataset name]/[data setname].py
This file contains the training dataset path.

## models/
### base/
This folder contains __stage 1__ training, also known as the joint training stage. It performs standard CNN training to obtain a basic feature extraction model.
The main training code of this stage is fscil_trainer.py. Please note that the mix parameter specified in the commands above defines the ratio of the mixup loss function, where 0<mix<1.

### cec/
This folder contains __stage 2__ training, also known as the pseudo incremental stage. This stage fine-tunes the model from the previous stage by continuously simulates new classes and trains them alongside old classes, utilizing the concept of meta-learning with multiple tasks.

The parameters specified in the commands above are listed below:
* __epochs_base__: number of epochs for the original method  
* __epochs_res__: number of epochs for the adjusted tasks in the new method. Typically, base is set to 0, and res is set to the desired number of training epochs 
* __episode_way__: number of classes to select for each task
* __episode_shot__: number of samples to select per class for each task
* __low_way__: number of classes to simulate for each task, which must be less than or equal to -episode_way
* __low_shot__: number of samples to simulate per class for the simulated classes in each task
* __episode_query__: number of samples required per class as query (test) for each task
* __pseudo_mode__: method for simulating new classes, with r representing rotation and c representing color jitter. The default is rotate








