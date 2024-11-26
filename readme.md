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









