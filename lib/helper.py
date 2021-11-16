import torch
from lib.fMRI import fMRIAutoencoderDataset, fMRI_Time_Subjs_Dataset
from lib.autoencoder import Encoder_basic, Decoder_Manifold, Decoder_Manifold_btlnk, Encoder_basic_btlnk
import numpy as np
from scipy.stats import zscore
import random
import pandas as pd
from sklearn.svm import SVC
import time
import multiprocessing as mp
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import logging
import os

def balance_train_data(X_train, Y_train, minimize=True):
    # assuming binary labels here
    (unique, counts) = np.unique(Y_train, return_counts=True)
    ind_max = np.where(counts == max(counts))
    ind_min = np.where(counts == min(counts))

    if len(counts[ind_max]) == 1 and counts[ind_max] / np.sum(counts) >= 0.52:
        where_max = np.where(Y_train == unique[ind_max])
        where_min = np.where(Y_train == unique[ind_min])
        min_indices = list(where_min[0])
        max_indices = list(where_max[0])
        if minimize:
            max_indices_subsamp = list(random.choices(where_max[0], k=len(min_indices)))
            train_indices = min_indices + max_indices_subsamp
        else:
            num2add = len(max_indices) - len(min_indices)
            min_indices_upsamp = list(random.choices(where_min[0], k=num2add)) + list(where_min[0])
            train_indices = min_indices_upsamp + max_indices
        train_indices.sort()

        X_train = X_train[train_indices]
        Y_train = Y_train[train_indices]
    return X_train, Y_train


def decode_timeseries(PARAMETERS):
    data, labels, balance_min, subject_id, n_folds = PARAMETERS
    """
    data: n_TR x hidden_dim
    labels: n_TR
    balance_min: None, True, False
    subject_id: 1--n_pt
    n_folds: kfold 
    """
    kf = KFold(n_splits = n_folds, shuffle=True)
    results = pd.DataFrame(columns=['subject_ID', 'fold', 'accuracy'])
    for split, (train, test) in enumerate(kf.split(np.arange(data.shape[0]))):
        model = SVC(kernel='rbf', C=10)
        X_train=data[train,:]
        y_train = labels[train]
        if balance_min is not None:
            X_train, y_train = balance_train_data(X_train, y_train, minimize=balance_min)
        model.fit(X_train, y_train)
        score = model.score(data[test,:], labels[test])
        results.loc[len(results)] = {'subject_ID':subject_id, 'fold':split, 'accuracy':score}

    return results


def drive_decoding(data_allpt, labels, kfold=5, balance_min = None, datasource='Sherlock', ROI='early_vis' ):
    t0=time.time()
    iterable_data = [(data_allpt[i], labels, balance_min, i+1, kfold) for i in range(len(data_allpt))]
    pool = mp.Pool(len(data_allpt))
    results = pool.map(decode_timeseries, iterable_data)
    results = pd.concat(results)
    # print('time consumption= ', time.time()-t0)
    return results
    

def extract_hidden_reps(encoder, decoders, dataset, device, amlps, args):
    # idx is pt-1 to index the mlp
    encoder.eval()
    for decoder in decoders:
        decoder.eval()
    if amlps is not None:
        for mlp in amlps:
            mlp.eval()

    hidden_reps = []
    aligned_hidden_reps = []
    for i in range(len(dataset)):
        input_TR = dataset[i]
        input_TR = torch.from_numpy(input_TR)
        input_TR = input_TR.unsqueeze(0).unsqueeze(0)
        input_TR = input_TR.float().to(device)
        if args.ae_type == 'conv':
            hidden, _,_ = encoder(input_TR)
        elif args.ae_type =='mlp':
            hidden = encoder(input_TR)
        elif args.ae_type =='mlp_md':
            common_hidden = encoder(input_TR)
            if len(decoders) == 1:
                decoder = decoders[0]
                decoder.eval()
                hidden, _ = decoder(common_hidden)
                aligned_hidden_reps.append(common_hidden.detach().cpu().numpy().flatten())
            else:
                pt = i//args.n_timerange
                decoders[pt].eval()
                hidden, _=decoders[pt](common_hidden)
                aligned_hidden_reps.append(common_hidden.detach().cpu().numpy().flatten())

        if amlps is not None:
            idx = i//args.n_timerange
            aligned_hidden = amlps[idx](hidden)
            aligned_hidden = aligned_hidden.detach().cpu().numpy().flatten()
            aligned_hidden_reps.append(aligned_hidden)


        hidden = hidden.detach().cpu().numpy().flatten()
        hidden_reps.append(hidden)
    hidden_reps = np.vstack(hidden_reps)
    if amlps or args.ae_type=='mlp_md':
        aligned_hidden_reps = np.vstack(aligned_hidden_reps)
    return hidden_reps, aligned_hidden_reps

def get_models(args):
    if args.symm:
        encoder = Encoder_basic_btlnk(args.input_size, args.hidden_dim * 4, args.hidden_dim * 2, args.hidden_dim, args.zdim)
    else:
        encoder = Encoder_basic(args.input_size, args.hidden_dim * 4, args.hidden_dim * 2, args.hidden_dim)

    decoders = []
    for i in range(args.n_subjects):
        if args.symm:
            decoder = Decoder_Manifold_btlnk(args.zdim, args.hidden_dim, args.hidden_dim * 2, args.hidden_dim * 4,
                                             args.input_size)
        else:
            decoder = Decoder_Manifold(args.zdim, args.hidden_dim, args.hidden_dim * 2, args.hidden_dim * 4,
                                       args.input_size)
        decoders.append(decoder)

    return encoder, decoders


def plot_losses(args, all_losses, len_dataloader, n_timepoints, savepath):

    fig, ax = plt.subplots()
    for j, l in enumerate(['total loss', 'reconstruction loss', 'manifold_reg_loss', 'regularization loss']):
        ax.scatter(range(args.load_epoch, args.n_epochs),
                   all_losses[:, j] * len_dataloader / n_timepoints / args.n_subjects, label=l)

    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    plt.yscale('log')
    plt.xscale('log')

    plt.show()
    plt.savefig(os.path.join(savepath, 'all_losses'))
    logging.info(f'Finished AE training {args.n_epochs} epochs')

def checkexist(targetdf, additional ):
    if len(targetdf)<1:
        return False
    for key in additional.keys():
        targetdf = targetdf.loc[targetdf[key] == additional[key]]
        if targetdf.shape[0]<1:
            return False
    return True