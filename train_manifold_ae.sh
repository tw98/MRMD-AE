#!/bin/bash

#SBATCH --partition=psych_day
#SBATCH --job-name=sherlock_cleaned
#SBATCH --output=results/sherlock_test.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tom.wallenstein@yale.edu
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --mem=50G

module load miniconda
conda activate tjl


ROI=early_visual
HIDDIM=64
ZDIM=20

BS=64
LAMBDA=0.01 #the common hidden lambda
MANILAM=2 # manifold embedding lambda
NEPOCHS=50
SAVEFREQ=1000
DATAP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/sherlock/MNI152_3mm_data/denoised_filtered_smoothed/ROI_data/${ROI}/data
DATANM=${ROI}_sherlock_movie.npy 
EMBEDP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/sherlock/MNI152_3mm_data/denoised_filtered_smoothed/ROI_data/${ROI}/embeddings
EMBEDNM=${ROI}_split0_${ZDIM}dimension_embedding_PHATE.npy
LABELP=/gpfs/milgram/scratch60/turk-browne/neuromanifold/sherlock/sherlock_labels_coded_expanded.csv

HALF=1
REG=mse
AE=mlp_md
#LR=1E-2
PTNUM=16 # only 16 patients now.
N_TR=1892
RUN_LABEL=sherlock_movie_${ROI}_${HIDDIM}_${ZDIM}_${LAMBDA}_${MANILAM}_${HALF}

SUMMARY_FILE=results/fMRI_manifoldAE_movie_result_summary.csv


python3 fMRI_Manifold_AE.py  --datapath=$DATAP --datanaming=$DATANM \
--embedpath=$EMBEDP --embednaming=$EMBEDNM \
--n_timerange=$N_TR --train_half=$HALF --hidden_dim=$HIDDIM --zdim=$ZDIM \
--batch_size=$BS --lam=$LAMBDA --lam_mani=$MANILAM --n_epochs=$NEPOCHS \
--reg=$REG --reg_ref \
--ae_type=$AE --shuffle_reg --n_subjects=$PTNUM \
--save_model --save_freq=$SAVEFREQ \
--downstream --labelpath=$LABELP \
--summary_file=$SUMMARY_FILE