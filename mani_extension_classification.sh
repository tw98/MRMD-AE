#!/bin/bash

##SBATCH --partition=psych_day
#SBATCH --job-name=mani_extension_compare
#SBATCH --output=results/mani_extension_compare.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tom.wallenstein@yale.edu
#SBATCH --time=23:59:00
#SBATCH -c 8
#SBATCH --mem=80G

module load miniconda
conda activate tjl

VOLSURF=MNI152_3mm_data
ZDIM=20
NTR=1976
NPT=16
LAMMANI=100
LAM=0.0

for ROI in early_visual
do
  for TRAINPCT in 10 30 50 70
  do
    for LAMMANI in 100
    do
      python3 mani_extension_classification.py --n_TR=$NTR --n_pt=$NPT --volsurf=$VOLSURF \
      --zdim=$ZDIM --lam=$LAM --lam_mani=$LAMMANI --train_percent=$TRAINPCT \
      --consecutive_time --oneAE
    done
  done
done
