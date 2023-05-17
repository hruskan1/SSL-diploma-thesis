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

# either specify all arguments you wish to change or do it by 
if (($# > 0))
then
    echo "Running conda run -n refmix python train_supervised_cityscape.py  $*"
    conda run -n refmix python train_supervised_cityscape.py  $*
else
    conda run -n refmix python train_supervised_cityscape.py -nt 100 --out ./cityscape_cat@sup100 --device 0 --BS 32
fi


