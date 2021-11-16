# check the classification accuracy of manifold extension by landmarking and AE

import warnings  # Ignore sklearn future warning
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import argparse
from lib.helper import checkexist,  extract_hidden_reps, drive_decoding
import os

parser = argparse.ArgumentParser()
parser.add_argument('--n_TR', type = int, default = 1976)
parser.add_argument('--train_percent', type=int, default=90)
parser.add_argument('--ROI', type = str, default = 'early_visual')
parser.add_argument('--hidden_dim', type = int, default = 64)
parser.add_argument('--zdim', type = int, default = 20)
parser.add_argument('--n_pt', type = int, default = 16)
parser.add_argument('--volsurf', type = str, default='MNI152_3mm_data') # default is volumetric data , alternatively fsaverage_data for surface data
parser.add_argument('--symm', action ='store_true') # use the symmetric config for encoder as decoder, so the latent encoder dim is the same as manifold dim
parser.add_argument('--lam', type = float, default = 0)
parser.add_argument('--lam_mani', type = float, default = 1)
parser.add_argument('--consecutive_time', action ='store_true', help='set active to make consecutive times e.g. 50% train will be first half of time series')
parser.add_argument('--oneAE', action = 'store_true', help='one encoder one decoder set up (vanilla AE or mr-AE)')


def get_decoding(embedpath, embednaming, labeldf, testTRs, args, embeds = None):
    entry = []
    test_embeds = []
    if embeds is not None:
        test_embeds = embeds
    else:
        for pt in range(1, args.n_pt+1):
            test_embeds.append(np.load(os.path.join(embedpath,f"sub-{pt:02}_{embednaming}")))
        test_embeds = np.vstack(test_embeds).reshape(args.n_pt, len(testTRs), -1)
    for labeltype in labeldf.columns.tolist()[2:]:
        labels = labeldf[labeltype].values[testTRs]
        results = drive_decoding(test_embeds, labels, balance_min=False)
        
        entry.append(results['accuracy'].mean())
        entry.append(results['accuracy'].std())
    return entry


def main():
    args = parser.parse_args()

    labelpath = '/gpfs/milgram/scratch60/turk-browne/neuromanifold/sherlock/binarized_sherlock_regressors.csv'
    labeldf = pd.read_csv(labelpath)

    path_trainTRs = f"/gpfs/milgram/scratch60/turk-browne/jh2752/data/sherlock_{args.train_percent}_trainTRs.npy"
    if args.consecutive_time:
        path_trainTRs = f"/gpfs/milgram/scratch60/turk-browne/jh2752/data/sherlock_{args.train_percent}_consec_trainTRs.npy"
        
    embednaming = f"{args.ROI}_{args.zdim}dimension_{args.train_percent}_test_PHATE.npy"
    embedpath = "/gpfs/milgram/scratch60/turk-browne/jh2752/data"

    savepath = f"/gpfs/milgram/scratch60/turk-browne/jh2752/results/sherlock_{args.volsurf}_{args.ROI}_mani_extend_{args.train_percent}"


    cols = ['train_percent', 'ROI', 'method', 'inout_mean', 'inout_std', 'music_mean', 'music_std']
    outdf = pd.DataFrame(columns=cols)

    trainTRs = np.load(path_trainTRs)
    testTRs = np.setxor1d(np.arange(args.n_TR), trainTRs)
    testTRs.sort()

    #landmark results:
    tr_pct = f"{args.train_percent}"
    print(f"---- Train TR Percentage {tr_pct}% ----")
    
    if args.consecutive_time:
        tr_pct = tr_pct+'_consec'
        embednaming = f"{args.ROI}_{args.zdim}dimension_{args.train_percent}_consec_test_PHATE.npy"
    entry = [tr_pct, args.ROI, 'PHATE_landmark']

    if not checkexist(outdf, dict(zip(cols[:3], entry))):
        print("get PHATE landmark decodings ...")
        entry_add = get_decoding(embedpath, embednaming, labeldf, testTRs, args)
        entry.extend(entry_add)
        outdf.loc[len(outdf)]=entry

    # mrmdAE extension
    if args.consecutive_time:
        savepath =savepath+'_consec'
    if args.oneAE:
        savepath = savepath + '_oneAE'
    mrmdAEfile =  f"mrmdAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}_testhidden.npy"
    if not os.path.exists(os.path.join(savepath, mrmdAEfile)):
        print(f'ERROR:{mrmdAEfile} not run yet ')
        
    else:
        embednaming = mrmdAEfile
        mrmd_embeds = np.load(os.path.join(savepath,mrmdAEfile))
        method = f'mrmdAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}'
        if args.oneAE:
            method = f'oneAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}'
        entry = [tr_pct, args.ROI, method]

        if not checkexist(outdf, dict(zip(cols[:3], entry))):
            print("get MR-AE decodings ...")
            entry_add = get_decoding(savepath, embednaming, labeldf, testTRs, args, embeds=mrmd_embeds)
            entry.extend(entry_add)
            outdf.loc[len(outdf)] = entry

    print(outdf)

if __name__=='__main__':
    main()
