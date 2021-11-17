import os
import pickle

import numpy as np
import scipy
from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt

import phate
import scprep
import graphtools as gt

import nibabel as nib
import nilearn.plotting
import nilearn.image
import nilearn.input_data

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

def power_of_2_ceil(x):
    return 2**int(np.ceil(np.log2(x)))

def expand_ROI_one_step(ROI_fdata):
    expanded_ROI = ROI_fdata.copy()
    for idx, x in np.ndenumerate(ROI_fdata):
        if x == 1:
            if idx[0] < ROI_fdata.shape[0] - 1:
                expanded_ROI[(idx[0] + 1, idx[1],     idx[2]    )] = 1
            if idx[0] > 0:
                expanded_ROI[(idx[0] - 1, idx[1],     idx[2]    )] = 1
            if idx[1] < ROI_fdata.shape[1] - 1:
                expanded_ROI[(idx[0]    , idx[1] + 1, idx[2]    )] = 1
            if idx[1] > 0:
                expanded_ROI[(idx[0]    , idx[1] - 1, idx[2]    )] = 1
            if idx[2] < ROI_fdata.shape[2] - 1:
                expanded_ROI[(idx[0]    , idx[1]    , idx[2] + 1)] = 1
            if idx[1] > 0:
                expanded_ROI[(idx[0]    , idx[1]    , idx[2] - 1)] = 1
    
    return expanded_ROI

def expand_ROI(ROI_fdata, n_steps):
    for i in range(n_steps):
        expanded_ROI = expand_ROI_one_step(ROI_fdata)
        ROI_fdata = expanded_ROI
        
    return expanded_ROI

def pad_timeseries(timeseries):
    """
    Pads the dims of an fMRI timeseries to the nearest powers of 2.
    Takes an input a timeseries with dims (x, y, z, time).
    """
    padded_x_dim = power_of_2_ceil(timeseries.shape[1])
    padded_y_dim = power_of_2_ceil(timeseries.shape[2])
    padded_z_dim = power_of_2_ceil(timeseries.shape[3])
    
    x_left_pad = int(np.floor((padded_x_dim - timeseries.shape[1]) / 2))
    x_right_pad = int(np.ceil((padded_x_dim - timeseries.shape[1]) / 2))
    y_left_pad = int(np.floor((padded_y_dim - timeseries.shape[2]) / 2))
    y_right_pad = int(np.ceil((padded_y_dim - timeseries.shape[2]) / 2))
    z_left_pad = int(np.floor((padded_z_dim - timeseries.shape[3]) / 2))
    z_right_pad = int(np.ceil((padded_z_dim - timeseries.shape[3]) / 2))
    
    timeseries = np.pad(timeseries,
                        pad_width=((0, 0),
                                   (x_left_pad, x_right_pad),
                                   (y_left_pad, y_right_pad),
                                   (z_left_pad, z_right_pad)),
                        constant_values=0)
    
    return timeseries

def pad_ROI_mask(ROI_mask):
    """
    Pads the dims of an fMRI ROI mask to the nearest powers of 2.  Takes an input a timeseries with dims (x, y, z, time).
    """ 
    
    padded_x_dim = power_of_2_ceil(ROI_mask.shape[0])
    padded_y_dim = power_of_2_ceil(ROI_mask.shape[1])
    padded_z_dim = power_of_2_ceil(ROI_mask.shape[2])
    
    x_left_pad = int(np.floor((padded_x_dim - ROI_mask.shape[0]) / 2))
    x_right_pad = int(np.ceil((padded_x_dim - ROI_mask.shape[0]) / 2))
    y_left_pad = int(np.floor((padded_y_dim - ROI_mask.shape[1]) / 2))
    y_right_pad = int(np.ceil((padded_y_dim - ROI_mask.shape[1]) / 2))
    z_left_pad = int(np.floor((padded_z_dim - ROI_mask.shape[2]) / 2))
    z_right_pad = int(np.ceil((padded_z_dim - ROI_mask.shape[2]) / 2))
    
    ROI_mask = np.pad(ROI_mask,
                      pad_width=((x_left_pad, x_right_pad),
                                 (y_left_pad, y_right_pad),
                                 (z_left_pad, z_right_pad)),
                      constant_values=0)
    
    return ROI_mask   

def generate_ROI_dataset(source_dir, target_dir, ROI_path, ROI_cluster=1, n_ROI_expansion_steps=5):
    """
    Converts the raw .nii timeseries files into cropped/padded
    .npy timeseries for a specific ROI. If the ROI contains multiple
    clusters this function will extract the one specified by the
    ROI_cluster parameter.
    """
    
    print("Generating mask...")
    ROI_nii = nib.load(ROI_path)
    ROI_expanded = expand_ROI(ROI_nii.get_fdata(), n_ROI_expansion_steps)
    cluster_labels, n_labels = scipy.ndimage.label(ROI_expanded)
    ROI_nii_cluster = nilearn.image.new_img_like(ROI_nii, cluster_labels == ROI_cluster)
    ROI_nii_cluster_cropped = nilearn.image.crop_img(ROI_nii_cluster, copy=False, pad=False)
    ROI_mask = ROI_nii_cluster_cropped.get_fdata()
    ROI_mask = pad_ROI_mask(ROI_mask)
    data_path = os.path.join(target_dir, "mask.npy")
    np.save(data_path, ROI_mask)
    
    patient_ids = range(1, 18)
    fmri_paths = []
    for patient_id in patient_ids:
        fmri_path = os.path.join(source_dir, f"sherlock_movie_s{patient_id}.nii")
        fmri_paths.append(fmri_path)
        
    for i, path in enumerate(fmri_paths):
        print(f"Generating patient{i + 1}.npy...")
        timeseries_nii = nib.load(path)
        timeseries_nii_cropped = nilearn.image.resample_img(timeseries_nii, 
                                                            target_affine=ROI_nii_cluster_cropped.affine, 
                                                            target_shape=ROI_nii_cluster_cropped.shape)
        
        
        timeseries = timeseries_nii_cropped.get_fdata()
        timeseries = np.transpose(timeseries, axes=(3, 0, 1, 2))
        timeseries = pad_timeseries(timeseries)
        timeseries *= ROI_mask
        
        data_path = os.path.join(target_dir, f"patient{i + 1}.npy")
        np.save(data_path, timeseries)
        
    print("Generating template_nii.pkl...")
    data_path = os.path.join(target_dir, f"template_nii.pkl")
    template_nii = nilearn.image.index_img(timeseries_nii_cropped, 0)
    pickle.dump(template_nii, open(data_path, 'wb'))

def generate_full_brain_dataset(source_dir, target_dir):
    """
    Converts the raw .nii timeseries files into cropped/padded
    full-brain .npy timeseries.
    """
    
    patient_ids = range(1, 18)
    fmri_paths = []
    for patient_id in patient_ids:
        fmri_path = os.path.join(source_dir, f"sherlock_movie_s{patient_id}.nii")
        fmri_paths.append(fmri_path)
        
    for i, path in enumerate(fmri_paths):
        print(f"Generating patient{i + 1}.npy...")
        timeseries_nii = nib.load(path)
        fmri = timeseries_nii.get_fdata()
        
        # The initial dimensions of each TR are 61 x 73 x 61
        # Here we crop the y dim and pad the x, z dims to make
        # each TR have dims 64 x 64 x 64
        fmri = fmri[:,4:-5,:,:]
        fmri = np.pad(fmri, pad_width=((2, 1), (0, 0), (2, 1), (0, 0)), constant_values=0) 
        data_path = os.path.join(target_dir, f"patient{i + 1}.npy")
        np.save(data_path, fmri)
        
    print("Generating template_nii.pkl...")
    data_path = os.path.join(target_dir, f"template_nii.pkl")
    template_nii = nilearn.image.index_img(timeseries_nii, 0)
    pickle.dump(template_nii, open(data_path, 'wb'))


class fMRIAutoencoderDataset(torch.utils.data.Dataset):
    
    def __init__(self, patient_ids, data_path, TRs, data_3d = True, data_name_suffix='early_visual_sherlock_movie.npy'):
        # This loads the entire dataset into memory. This is not 
        # recommended for training neural networks but it is faster.
        # Training with the entire full-brain dataset takes ~140GB
        # of memory.
        timeseries = []
        for patient_id in patient_ids:
            if data_3d:
                path = os.path.join(data_path, f"patient{patient_id}.npy")
                timeseries.append(np.load(path)[TRs])
            else:
                path = os.path.join(data_path, f"sub-{patient_id:02}_{data_name_suffix}")
                timeseries.append(np.load(path)[TRs])
            
        self.timeseries = np.concatenate(timeseries, axis=0)

        self.TRs = list(TRs)
        self.patient_ids = list(patient_ids)
        self.data_3d = data_3d
        
    def __len__(self):
        return self.timeseries.shape[0]
    
    def get_TR_dims(self):
        return self.timeseries.shape[1:]
    
    def visualize_input_TR(self, patient_id, TR):
        try:
            TR_index = self.TRs.index(TR)
        except ValueError as e:
            raise ValueError(f"TR {TR} not found in dataset.") from e
            
        try:
            patient_index = self.patient_ids.index(patient_id)
        except ValueError as e:
            raise ValueError(f"Patient {patient_id} not found in dataset.") from e
            
        timeseries_index = patient_index * len(self.TRs) + TR_index
        input_TR_img = nilearn.image.new_img_like(self.template_nii, self.timeseries[timeseries_index])
        nilearn.plotting.plot_stat_map(input_TR_img)
        
    def visualize_output_TR(self, patient_id, TR, model, device):
        try:
            TR_index = self.TRs.index(TR)
        except ValueError as e:
            raise ValueError(f"TR {TR} not found in dataset.") from e
            
        try:
            patient_index = self.patient_ids.index(patient_id)
        except ValueError as e:
            raise ValueError(f"Patient {patient_id} not found in dataset.") from e
            
        timeseries_index = patient_index * len(self.TRs) + TR_index
        input_TR = self.timeseries[timeseries_index]
        input_TR = torch.from_numpy(input_TR)
        input_TR = input_TR.unsqueeze(0).unsqueeze(0)
        input_TR = input_TR.float().to(device)
        output_TR, _ = model(input_TR)
        output_TR = output_TR.detach().cpu().numpy()[0, 0, :, :, :]
        output_TR *= self.mask
        output_TR_img = nilearn.image.new_img_like(self.template_nii, output_TR)
        nilearn.plotting.plot_stat_map(output_TR_img)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        return self.timeseries[idx]


class fMRI_Time_Subjs_Dataset(torch.utils.data.Dataset):
    # stores multipatient time series in shape T x m x HWD or Txm (1d spanned)
    # T: time length, m: num of patients, HWD voxels in 3d HxWxD
    # when batch, fixed timepoint batch for all patients

    def __init__(self, patient_ids, data_path, TRs, data_3d = True, fnbase = ''):
        # This loads the entire dataset into memory. This is not
        # recommended for training neural networks but it is faster.
        # Training with the entire full-brain dataset takes ~140GB
        # of memory.
        timeseries = []
        for patient_id in patient_ids:
            print(f"loading data from patient {patient_id} into memory")
            if data_3d:
                path = os.path.join(data_path, f"patient{patient_id}.npy")
                timeseries.append(np.load(path)[TRs])
            else:
                path = os.path.join(data_path, f"sub-{patient_id:02}_early_visual_sherlock_movie.npy")
                if len(fnbase)>1:
                    path = os.path.join(data_path, f"sub-{patient_id:02}_"+fnbase)
                timeseries.append(np.load(path)[TRs])
            
        timeseries = np.concatenate(timeseries, axis=0)

        # here separates if prepare dataset for autoencoder with 3d convolutional layer or mlp
        if data_3d:
            timeseries = np.reshape(timeseries, [len(patient_ids), int(timeseries.shape[0] / len(patient_ids)),
                                             timeseries.shape[1], timeseries.shape[2], timeseries.shape[3]]) # shape [m xT x HWD]
            path = os.path.join(data_path,
                                f"template_nii.pkl")  # not 3d data no need for template or mask
            self.template_nii = pickle.load(open(path, 'rb'))

            path = os.path.join(data_path, f"mask.npy")
            self.mask = np.load(path)

        else:
            timeseries = np.reshape(timeseries, [len(patient_ids), int(timeseries.shape[0] / len(patient_ids)), -1])
            # shape m x T
        self.timeseries = np.swapaxes(timeseries, 0, 1)  # shape [T x m x HWD]



        self.TRs = list(TRs)
        self.patient_ids = list(patient_ids)

    def __len__(self):
        return self.timeseries.shape[0] # in this dataset length is number of TRs

    def get_TR_dims(self):
        return self.timeseries.shape[2:] # HxWxD one timepoint voxels 3d shape

    def visualize_input_TR(self, patient_id, TR):
        try:
            TR_index = self.TRs.index(TR)
        except ValueError as e:
            raise ValueError(f"TR {TR} not found in dataset.") from e

        try:
            patient_index = self.patient_ids.index(patient_id)
        except ValueError as e:
            raise ValueError(f"Patient {patient_id} not found in dataset.") from e

        input_TR_img = nilearn.image.new_img_like(self.template_nii, self.timeseries[TR_index, patient_index])
        nilearn.plotting.plot_stat_map(input_TR_img)


    def __getitem__(self, idx): # TODO: is getitem used anywhere?
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.timeseries[idx]

class fMRI_Time_Subjs_Embed_Dataset(torch.utils.data.Dataset):
    # stores multipatient time series in shape T x m x HWD or Txm (1d spanned)
    #        and the manifold embeddings of of the same time-patient series.
    # T: time length, m: num of patients, HWD voxels in 3d HxWxD
    # when batch, fixed timepoint batch for all patients
    def __init__(self, patient_ids, data_path, embed_path, TRs,
                 emb_name_suffix = 'early_visual_SRM_10dimensions_PHATE.npy',data_3d=True, data_name_suffix='early_visual_sherlock_movie.npy'):
        # This loads the entire dataset into memory. This is not
        # recommended for training neural networks but it is faster.
        # Training with the entire full-brain dataset takes ~140GB
        # of memory.
        timeseries = []
        embeds = []
        for patient_id in patient_ids:
            print(f"loading data from patient {patient_id} into memory")
            if data_3d:
                path = os.path.join(data_path, f"patient{patient_id}.npy")
                timeseries.append(np.load(path)[TRs])
            else:
                path = os.path.join(data_path, f"sub-{patient_id:02}_{data_name_suffix}")
                timeseries.append(np.load(path)[TRs])

        timeseries = np.concatenate(timeseries, axis=0)

        # here separates if prepare dataset for autoencoder with 3d convolutional layer or mlp
        if data_3d:
            timeseries = np.reshape(timeseries, [len(patient_ids), int(timeseries.shape[0] / len(patient_ids)),
                                                 timeseries.shape[1], timeseries.shape[2],
                                                 timeseries.shape[3]])  # shape [m xT x HWD]
            path = os.path.join(data_path,
                                f"template_nii.pkl")  # not 3d data no need for template or mask
            self.template_nii = pickle.load(open(path, 'rb'))

            path = os.path.join(data_path, f"mask.npy")
            self.mask = np.load(path)

        else:
            timeseries = np.reshape(timeseries, [len(patient_ids), int(timeseries.shape[0] / len(patient_ids)), -1])
            print(f"input data shape: {timeseries.shape}")
            # shape m x T

        for patient_id in patient_ids:
            path = os.path.join(embed_path, f"sub-{patient_id:02}_{emb_name_suffix}")
            # embeds.append(np.load(path)[TRs]) # The new embedding is in splits
            embeds.append(np.load(path))
        embeds = np.concatenate(embeds, axis=0)
        print('read in embeddings, shape before reshaping', embeds.shape )
        embeds = np.reshape(embeds, [len(patient_ids), int(embeds.shape[0]/len(patient_ids)), -1])
        print(f"embedding data shape: {embeds.shape}")


        self.timeseries = np.swapaxes(timeseries, 0, 1)  # shape [T x m x HWD]
        self.embeds = np.swapaxes(embeds, 0,1)
        self.TRs = list(TRs)
        self.patient_ids = list(patient_ids)

    def __len__(self):
        return self.timeseries.shape[0] # in this dataset length is number of TRs

    def get_TR_dims(self):
        return self.timeseries.shape[2:] # HxWxD one timepoint voxels 3d shape

    def get_embed_dims(self):
        return self.embeds.shape[2]

    def visualize_input_TR(self, patient_id, TR):
        try:
            TR_index = self.TRs.index(TR)
        except ValueError as e:
            raise ValueError(f"TR {TR} not found in dataset.") from e

        try:
            patient_index = self.patient_ids.index(patient_id)
        except ValueError as e:
            raise ValueError(f"Patient {patient_id} not found in dataset.") from e

        input_TR_img = nilearn.image.new_img_like(self.template_nii, self.timeseries[TR_index, patient_index])
        nilearn.plotting.plot_stat_map(input_TR_img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.timeseries[idx], self.embeds[idx]