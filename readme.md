# FFCIL
This repository contains the code release for the FFCIL paper titled Fine-grained Few-shot Class Incremental Learning with Destruction-construction Integration for Electronic Display Defect Detection [link].

# Environment
A conda environment named auofscil can be created with the following command:

    conda env create -f environment.yml

# Dataset
Public datasets supporting the experiments can be downloaded from the following links:
* CUB_200_2011 [https://www.vision.caltech.edu/datasets/cub_200_2011/]
* Stanford Dogs [https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset]

# Training
The FFCIL training can be performed in three steps. 
1.  Train the DCL model (https://doi.org/10.1109/CVPR.2019.00530) using the code provided at https://github.com/JDAI-CV/DCL on your chosen dataset (either CUB200 or Stanford Dogs).
2.  Load the weights from Step 1 (placeholder name: DCLTrainedModelFile.pth) and execute the following command to continue with the remaining pretraining stage:

           python train.py -project base -dataset datasetName -base_mode 'ft_cos' -new_mode 'avg_cos' -lr_base 0.001 -lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Cosine -gpu 0 -temperature 16 -mix 0.5 -model_dir DCLTrainedModelFile.pth -batch_size_base 64 -batch_size_new 512

4.  Load the weights from Step 2 (placeholder name: pretrain.pth) and execute the following command below to proceed with pseudo-incremental and incremental stage training.

           python train.py -project cec -dataset datasetName -epochs_base 0 -epochs_res 100 -episode_way 5 -episode_shot 1 -low_way 5 -low_shot 1 -episode_query 10 -lr_base 0.002 -lrg 0.001 -step 20 -gpu 0,1 -model_dir params/pretrain.pth

#### Please note that the `datasetName` __MUST__ be replaced with either cub200 or stanford_dogs ####

# Testing
The FFCIL testing is integrated into the code for Step 4, which involves calculating the testing data for each task.
The testing phase provides accuracy metrics for each task, including:
* `all_acc` (equivalent to _acc<sub>all</sub>_), which measures the accuracy across all classes within a task
* `base_acc` (equivalent to _acc<sub>prm</sub>_) which measures the accuracy of the classes from previous task
* `new_acc` (equivalent to _acc<sub>add</sub>_) which measures the accuracy of the new classes introduced in the current task

# Files
There are three main folders in this project.
* `data/index_list` contains the indices of image files for each tasks across all datasets.
* `dataloader` contains the image loading and preprocessing for both training and testing phases.
* `model` contains the main function for managing the training and testing processes. It includes the following subfolders:
    * `base` handles the pretraining stage.
    * `cec` handles both pseudo-incremental and incremental learning stages.


