import warnings  # Ignore sklearn future warning
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch
import logging
from lib.fMRI import fMRIAutoencoderDataset
import os
from lib.helper import extract_hidden_reps, drive_decoding, get_models
from lib.downstream_measures import time_segment_matching, ISC
import pandas as pd


def downstream_analysis(args, param, device, encoder, decoders):
    # record all results

    cols = ['data', 'n_subject', 'train_half', 'common_hiddendim','manifold_embed_dim',
            'common_latent_regularization', 'lambda_common', 'manifold_regularization', 
            'lambda_mani', 'manifold_type', 'ROI', 'HEM', 'encoder', 'decoder', 'reg_shuffle', 
            'load_path', 'load_epoch', 'lam_trans'
            ]

    datasource = '/'.join(args.embedpath.split('/')[-6:-1])

    entry = [
        datasource, args.n_subject, args.train_half, args.hidden_dim, args.zdim,
        args.reg, args.lam, 'mse', args.lam_mani,
        param.manifoldtype, param.roi, param.hemisphere,
        args.ae_type, args.ae_type, args.shuffle_reg, param.chkpt_savepath, args.n_epochs,
        args.lam_xsubj
        ]

    if os.path.exists(args.summary_file):
        resultsdf = pd.read_csv(args.summary_file)
        resultsdf.loc[len(resultsdf)] = dict(zip(cols, entry))
    else:
        resultsdf = pd.DataFrame([entry], columns=cols)

    args.loadpath = param.chkpt_savepath
    args.load_epoch = args.n_epochs
    print('extract hidden rep and calculate metrics and produce phate plots')
    
    if args.train_half is not None:
        # cross-validate the other half. e.g. trained model on the first half data, then apply the trained model on the second half
        if args.train_half == 2:
            train_half = np.arange(args.n_timerange)
        else:
            train_half = np.arange(args.n_timerange, 2 * args.n_timerange)
    else:
        train_half = np.arange(args.n_timerange)

    logging.info(f'apply trained model on {args.train_half} half to data[{train_half[0]}-{train_half[-1]}]')

    # load dataset
    if 'StudyForrest' in args.datapath:
        param.patient_ids = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 17, 18, 19, 20]
    else:
        param.patient_ids = np.arange(1, args.n_subject + 1)

    dataset = fMRIAutoencoderDataset(param.patient_ids,
                                    args.datapath,
                                    train_half,
                                    data_3d=param.data_3d,
                                    data_name_suffix=args.datanaming)

    if len(dataset) != args.n_timerange * args.n_subject:
        print('Error: dataset timepoints not equal to desired timepoints.')

    # continue to use the last state of encoder and decoder from training so that no need to load the saved checkpoints.
    latent_mlps = None

    encoder.eval()
    hidden, al_hidden = extract_hidden_reps(encoder, decoders, dataset, device, latent_mlps, args)
    hidden = hidden.reshape(args.n_subject, args.n_timerange, -1) # hidden is all patients, and interested timepoints

    if args.amlp or args.ae_type == 'mlp_md':
        al_hidden = al_hidden.reshape(args.n_subject, args.n_timerange, -1)

    print('hidden representation shape: ',hidden.shape)
    datatype = os.path.split(args.datapath)[-1]
    modeldata = os.path.split(args.loadpath)[-1]
    saveto = os.path.join(args.loadpath,
                            f'{datatype}_{modeldata}_model_on_otherhalf_data_e{args.load_epoch}.npy')

    np.save(saveto, hidden)
    resultsdf.loc[len(resultsdf)-1, 'hiddenfile']=saveto
    print('saved extracted latent representation to : ', saveto)
    logging.info (f'saved extracted latent representation to : {saveto}')
    tsm = np.nan
    isc = np.nan
    tsm_std = np.nan
    isc_std = np.nan
    if args.amlp or args.ae_type == 'mlp_md': # the amlp in mlp_md case is the common embedding
        saveto = os.path.join(args.loadpath,
                                f'{datatype}_{modeldata}_model_on_{args.train_half}half_data_e{args.load_epoch}_amlp.npy')
        np.save(saveto, al_hidden)
        resultsdf.loc[len(resultsdf) - 1, 'aligned_hiddenfile'] = saveto
        print('amlp hidden saved to: ', saveto)
        logging.info(f"aligned encoder latent representation saved to: {saveto}")

        tsm = time_segment_matching(al_hidden)
        tsm_std = tsm.std()
        tsm = tsm.mean()

        isc = ISC(al_hidden)
        isc_std = isc.std()
        isc = isc.mean()

    resultsdf.loc[len(resultsdf)-1, 'tsm_mean' ]=tsm
    resultsdf.loc[len(resultsdf) - 1, 'tsm_std']=tsm_std
    resultsdf.loc[len(resultsdf) - 1, 'isc_mean']=isc
    resultsdf.loc[len(resultsdf) - 1, 'isc_std']=isc_std

    # SVC prediction
    if args.labelpath is None:
        logging.info('ERROR: need label to perform classification. set labelpath=')
    else:
        labeldf = pd.read_csv(args.labelpath)
        accuracies = dict()
        for labeltype in labeldf.columns.tolist()[2:]:
            labels = labeldf[labeltype].values[train_half]
            results = drive_decoding(hidden, labels, balance_min = False) #augment train data
            accuracies[labeltype] = results
            resultsdf.loc[len(resultsdf)-1, labeltype+'_acc']=results['accuracy'].mean()
            results.to_csv(os.path.join(args.loadpath,
                f"{(args.summary_file.split('.')[0]).split('/')[-1]}_row_{len(resultsdf)-1}_{labeltype}_accuracies.csv"),
                            index=False)

    resultsdf.to_csv(args.summary_file, index=False)