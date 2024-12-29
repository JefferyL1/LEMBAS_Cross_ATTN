#!/bin/bash
#SBATCH -J LEMBAS_w_attn_VCAP
#SBATCH -o LEMBAS_w_attn_VCAP%j.out
#SBATCH -e LEMBAS_w_attn_VCAP%j.err
#SBATCH --mail-user=jefferyl@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --exclusive

. /nobackup/users/jefferyl/miniforge3/etc/profile.d/conda.sh

conda activate ml_env

python /nobackup/users/jefferyl/LauffenLab/LEMBAS_w_attn/code/training.py


