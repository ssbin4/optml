#!/bin/sh
#SBATCH -J  CARS196          # Job name
#SBATCH -o  cars_196.%j.out    # Name of stdout output file (%j expands to %jobId)
#SBATCH -t 1-00:00:00        # Run time (hh:mm:ss) 

#### Select  GPU
#SBATCH -p titanxp           # queue  name  or  partiton
#SBATCH   --gres=gpu:2           # gpus per node##  node 지정하기
#SBATCH   --nodes=1              # number of nodes
#SBATCH   --ntasks-per-node=1
#SBATCH   --cpus-per-task=1

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge
module load postech

##  Python  Virtual Env ##    ⇒> 가상환경

echo "Start"
echo "condaPATH"

echo "source $HOME/anaconda3/etc/profile.d/conda.sh"
source $HOME/anaconda3/etc/profile.d/conda.sh    #경로

echo "conda activate cbm"
conda activate cbm    #사용할 conda env
python run_resnet_training_cli.py --path_images ./CARS196/cars_train --path_data ./CARS196/devkit/cars_train_annos.mat --path_labels ./CARS196/devkit/cars_meta.mat --path_model_checkpoint ./checkpoint/ --checkpoint_frequency 50 --model ResNet18 --epochs 1000 --validation_frequency 1  --number_of_classes 196 --data_subset 1.0 --batch_size 32 --weight_decay 0.01 --lr_scheduler step  --annealing_factor 0.1 --momentum 0.9 --init_lr 0.001 --scheduler_rate 10 --no-nesterov --no-freeze_weights
date

echo " condadeactivate tf-gpu-py36"

conda deactivate #마무리 deactivate

squeue--job $SLURM_JOBID

echo  "##### END #####"