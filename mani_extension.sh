#!/bin/bash
conda activate env_fmri

ROI=early_visual # region of interest
BS=64 # batch size
NEPOCHS=100 # number of epochs
SAVEFREQ=100 # checkpoint save frequency
HIDDIM=64 # common latent layer dimension
ZDIM=20 # hidden dimension
LAMMANI=100  # manifold regularization
LAM=0 # commom embedding layer regularization

for TRAINPCT in 50 30 10 70
 do
    python3 mani_extension.py --train_percent=$TRAINPCT --ROI=$ROI \
    --hidden_dim=$HIDDIM --zdim=$ZDIM --volsurf=$VOLSURF \
    --batch_size=$BS --lam=$LAM --lam_mani=$LAMMANI \
    --consecutive_time # only when we want the train-test time to be consecutive
 done
done