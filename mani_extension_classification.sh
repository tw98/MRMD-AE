#!/bin/bash

conda activate env_fmri

LAMMANI=100 # manifold layer regularization
LAM=0.0 # common embedding layer regularization

for ROI in early_visual
do
  for TRAINPCT in 10 30 50 70
  do
    python3 mani_extension_classification.py \
    --lam=$LAM --lam_mani=$LAMMANI --train_percent=$TRAINPCT \
    --consecutive_time --oneAE
  done
done
