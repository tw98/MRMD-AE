#!/bin/bash
conda activate env_fmri

ROI=early_visual # region of interest
HIDDIM=64 # common latent layer dimension
ZDIM=20 # manifold layer dimension
BS=64 # batch size
LAMBDA=0.01 # common hidden lambda
MANILAM=0.01 # manifold embedding lambda
NEPOCHS=4000 # number of epochs
SAVEFREQ=1000 # checkpoint save frequency

HALF=1 # train half (1->first half, 2->second half)
EMBEDNM=${ROI}_split0_${ZDIM}dimension_embedding_PHATE.npy # embedding file (split0 corresponds to first half, split1 -> second half)

SUMMARY_FILE=results/fMRI_manifoldAE_movie_result_summary.csv # summary of results

python3 fMRI_Manifold_AE.py  --train_half=$HALF --embednaming=$EMBEDNM \
--hidden_dim=$HIDDIM --zdim=$ZDIM \
--batch_size=$BS --lam=$LAMBDA --lam_mani=$MANILAM --n_epochs=$NEPOCHS \
--reg_ref --shuffle_reg \
--save_model --save_freq=$SAVEFREQ \
--downstream --summary_file=$SUMMARY_FILE