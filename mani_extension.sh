#!/bin/bash
conda activate env_fmri

BS=64
NEPOCHS=100
SAVEFREQ=100
PTNUM=16 # only 16 patients now.
VOLSURF=MNI152_3mm_data #  for surface data or MNI152_3mm_data for volumetric data
HIDDIM=64
ZDIM=20
EMBED=PHATE
LAMMANI=10
LAM=0 #

#for ROI in early_visual #early_visual #aud_early pmc_nn
#do
#  for TRAINPCT in 50 30 10 70 90
#  do
#    for LAMMANI in 0.01 1 100 500 1000
#    do
#      python3 mani_extension.py --train_percent=$TRAINPCT --ROI=$ROI \
#      --hidden_dim=$HIDDIM --zdim=$ZDIM --volsurf=$VOLSURF \
#      --batch_size=$BS --lam=$LAM --lam_mani=$LAMMANI \
#      --consecutive_time # only when we want the train-test time to be consecutive
#    done
#  done
#done

# Following for individual mr-AE
LAMMANI=100
for ROI in early_visual # aud_early pmc_nn
do
  for TRAINPCT in 70 50 30 10
  do
    for PT in {1..16}
    do
      python3 mani_extension.py --train_percent=$TRAINPCT --ROI=$ROI \
      --hidden_dim=$HIDDIM --zdim=$ZDIM --volsurf=$VOLSURF \
      --n_epochs=$NEPOCHS \
      --batch_size=$BS --lam=$LAM --lam_mani=$LAMMANI \
      --consecutive_time \
      --oneAE
#      --ind_mrAE --pt=$PT \
    done
  done
done