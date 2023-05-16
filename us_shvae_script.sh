#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:55:00
#SBATCH --mem-per-cpu=5G
#SBATCH --mail-user=hruskan1@fel.cvut.cz
#SBATCH --mail-type=ALL
#SBATCH --output=output-%j.out
### <SBATCH--chdir=/mnt/personal/hruskan1/exp>


if (($# > 0))
then
    conda run -n refmix python shvae_cityscape_train_unsup.py -c $1
else
    conda run -n refmix python shvae_cityscape_train_unsup.py -c shvae-unsup.yaml
fi


