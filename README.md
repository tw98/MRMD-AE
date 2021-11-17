# ICLR 2022: LEARNING SHARED NEURAL MANIFOLDS FROM MULTI-SUBJECT FMRI DATA

This codebase has the purpose to allow readers to reproduce our experiments. 

## Requirements/Setup

To run the scripts, a number of packages need to be installed. We used Anaconda as package manager. 

To install a suitable environment with all necessary packages, run:


```
conda create --name env_fmri --file spec-file.txt
```

## Data

For this demonstration, we will use our preprocessed data files from the [Sherlock Movie Watching Dataset](http://arks.princeton.edu/ark:/88435/dsp01nz8062179) (*Chen et. al (2017)*, ). We had to apply a number of preprocessing steps to make the raw data applicable to our experiment. First, we did the standard fMRI preprocessing steps. Next, the functional images from the dataset were aligned to the common MNI template brain based on anatomical landmarks. In this anatomical template, we defined several regions of interest (ROIs) to target in our analyses: the early visual cortex, late visual cortex, early auditory cortex, and the posterior medial cortex (PMC). We selected these regions as they robustly respond to audiovisual stimuli (early visual, late visual, and early auditory) and PMC was targeted in the original *Sherlock* analyses for its memory involvement (Chen et. al (2017)). We extracted timeseries data from the voxels in each of these regions (early visual: 307 voxels, late visual: 1008 voxels, early auditory: 1018 voxels, PMC: 481 voxels) and collapsed the spatial structure within this ROI into a [timepoints, voxels] matrix of activity for each voxel across time. Within this matrix, we z-scored each voxel across its timeseries to account for differences across voxels in mean activation.

One can download the data, all necessary PHATE embeddings, and pre-trained models under this link (). To make the distribution of the code and data easier, we have only shared data for the early visual ROI. 

## Code

The codebase contains the scripts to reproduce the results for figure 2 and table 1. Moreover, we provide an example to extract the embedding of a new subject after training the MRMD-AE with multiple different subject (i.e. train with the first 15 subjects, extract embedding for the 16th subject.)

### (1) Manifold Extension Results (Figure 2)

To train the autoencoder for the manifold extension task, run:

```
source mani_extension.sh
```

This will train an autoencoder model for each condition, i.e. percentage of dataset used. 

Since the training can take a long time depending on machine, we have provided pre-trained models and already computed embeddings in the data folder, one can use for the subsequent analysis. If one wants to do all computations newly, one should set the --new flag in the two scripts explained below. 

To do the MSE comparison between the manifold extended coordinates and the ground truth coordinates for each experiment condition, run:

```
source mani_extension_compare.sh
```

Similarly, to perform the classification task, run:

```
source mani_extension_classification.sh
```

### (2) Classification Results

To reproduce the results for the within-subject manifold classification, use. 

```
source fMRI_Manifold_AE.sh 
```

This will train an autoencoder and perform the classification task (only if, the --downstream flag is set)  

### (3) Embedding for New Subjects

The script *mdm_AE_xsubj.py* extracts the embedding for a new test subject after training the MRMD-AE on the remaining subjects. 

In the example below, the MRMD-AE is trained on the data of the first 15 subjects, and the test subject is the 16th subject. 

```
python mdm_AE_xsubj.py --testpt=16 --embednaming=early_visual_sherlock_movie_20dimension_embedding_PHATE.npy --shuffle_reg
```



