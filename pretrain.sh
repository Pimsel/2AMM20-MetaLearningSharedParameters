#!/bin/bash

#SBATCH --time=5:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --tasks-per-node 4
#SBATCH --gpus=1
#SBATCH --output=R-%x.%j.out


module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1


source ~/2AMM20_NFgroup1/bin/activate


wandblogin="$(<../wandb.login)"
wandb login "$wandblogin"


python pretrain.py