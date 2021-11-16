# compare the manifold extension calculated in 2 ways to the ground truth
# ground truth: phate from the full datasets for each subject
# 1. test manifold hidden from AE
# 2. phate landmark extension

import warnings  # Ignore sklearn future warning
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import argparse
import os
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
import phate
import time
from lib.helper import checkexist

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
parser.add_argument('--ind_mrAE', action='store_true', help='set active to compare independent MR-AE')
parser.add_argument('--oneAE', action = 'store_true', help='one encoder one decoder set up (vanilla AE or mr-AE)')


def get_metric(dist_matrix):
    # dist_matrix: n_pt x testTRs x trainTRs
    per_sub_mean = dist_matrix.mean(axis=(1)).reshape(dist_matrix.shape[0],1,dist_matrix.shape[2])
    dists = np.linalg.norm(dist_matrix-per_sub_mean , axis =2) # this is the distance of each testTR to the center of the test-train matrix
                                                            #shape is n_pt x testTRs
    average_dist  = dists.mean() # subject average mean distance
    dist_std = dists.std(axis=1).mean() # subject average distance std
    return [average_dist, dist_std]

def main():
    args = parser.parse_args()

    datapath = f"/gpfs/milgram/scratch60/turk-browne/neuromanifold/sherlock/{args.volsurf}/denoised_filtered_smoothed/ROI_data/{args.ROI}/data"
    datanaming = f"{args.ROI}_sherlock_movie.npy"
    truephatepath = "/gpfs/milgram/scratch60/turk-browne/jh2752/data"
    truephatenaming = f"{args.ROI}_{args.zdim}dimension_PHATE.npy"
    savepath = f"/gpfs/milgram/scratch60/turk-browne/jh2752/results/sherlock_{args.volsurf}_{args.ROI}_mani_extend_{args.train_percent}"
    embednaming = f"{args.ROI}_{args.zdim}dimension_{args.train_percent}_train_PHATE.npy"

    path_trainTRs = f"/gpfs/milgram/scratch60/turk-browne/jh2752/data/sherlock_{args.train_percent}_trainTRs.npy"
    if args.consecutive_time:
        path_trainTRs = f"/gpfs/milgram/scratch60/turk-browne/jh2752/data/sherlock_{args.train_percent}_consec_trainTRs.npy"

    
    # check if PHATE ground truth manifold has been computed before
    if not os.path.exists(os.path.join(truephatepath,f"sub-01_{truephatenaming}")):
        print('prepare ground truth phate embeddings')
        for pt in range(1, args.n_pt+1):
            X = np.load(os.path.join(datapath, f"sub-{pt:02}_{datanaming}"))
            pop = phate.PHATE(n_components =args.zdim)
            X_p = pop.fit_transform(X)
            np.save(os.path.join(truephatepath, f"sub-{pt:02}_{truephatenaming}"), X_p)


    print(f'---- TRAIN PERCENTAGE {args.train_percent}% ----')
    tr_pct = args.train_percent
    if args.consecutive_time:
        tr_pct = f"{args.train_percent}_consec"


    trainTRs = np.load(path_trainTRs)
    testTRs = np.setxor1d(np.arange(args.n_TR), trainTRs)
    testTRs.sort()

    # get the test TRs manifold euc distance to each of the other
    truedists =[]
    for pt in range(1, args.n_pt+1):
        X_p = np.load(os.path.join(truephatepath, f"sub-{pt:02}_{truephatenaming}"))
        truedists.append(euclidean_distances(X_p[testTRs], X_p[trainTRs]))
    truedists = np.vstack(truedists).reshape(args.n_pt, len(testTRs), len(trainTRs))

    cols = ['train_percent', 'ROI', 'method', 'dist_mse_mean', 'dist_mse_std']
    outdf = pd.DataFrame(columns = cols)

    # check if the landmark has run:  
    entry = [tr_pct, args.ROI, 'PHATE_landmark']
    exist = checkexist(outdf, dict(zip(cols[:3], entry)))

    if not exist:
        lmdists =[] # the phate landmark version

        if args.consecutive_time:
            embednaming = f"{args.ROI}_{args.zdim}dimension_{args.train_percent}_consec_train_PHATE.npy"
            
        for pt in range(1, args.n_pt+1):
            p_train = np.load(os.path.join(truephatepath, f"sub-{pt:02}_{embednaming}"))
            testphate_file = f"sub-{pt:02}_{embednaming}"
            testphate_file = testphate_file.replace('_train_', '_test_')
            p_test = np.load(os.path.join(truephatepath, testphate_file))
            lmdists.append(euclidean_distances(p_test, p_train))
        lmdists = np.vstack(lmdists).reshape(args.n_pt, len(testTRs), len(trainTRs))
        
        mse_lm = []
        for pt in range(truedists.shape[0]):
            mse_lm.append(mean_squared_error(truedists[pt], lmdists[pt]))

        entry = [args.train_percent, args.ROI, 'PHATE_landmark', np.mean(mse_lm), np.std(mse_lm)]
        if args.consecutive_time:
            entry = [f"{args.train_percent}_consec", args.ROI, 'PHATE_landmark', np.mean(mse_lm), np.std(mse_lm)]

        if not checkexist(outdf, dict(zip(cols[:3],entry[:3]))):
            outdf.loc[len(outdf)]=entry

    else: 
        print(f"{entry} has run")

    if not args.ind_mrAE:
        # mrmdAE extension or oneAE extension
        if args.consecutive_time:
            savepath =savepath+'_consec'
        if args.oneAE:
            savepath = savepath + '_oneAE'
        mrmdAEfile =  f"mrmdAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}_testhidden.npy"
        
        if not os.path.exists(os.path.join(savepath, mrmdAEfile)):
            print(f'ERROR:{mrmdAEfile} not run yet ')
        else:
            p_test = np.load(os.path.join(savepath,
                                          f"mrmdAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}_testhidden.npy"))
            mrmddists=[]
            for pt in range(1, args.n_pt+1):
                p_train = np.load(os.path.join(truephatepath,
                                               f"sub-{pt:02}_{embednaming}"))
                mrmddists.append(euclidean_distances(p_test[pt-1], p_train ))
            mrmddists=np.vstack(mrmddists).reshape(args.n_pt, len(testTRs), len(trainTRs))
            
            mse_mrmd = []
            for pt in range(truedists.shape[0]):
                mse_mrmd.append(mean_squared_error(truedists[pt], mrmddists[pt]))

            if args.oneAE:
                method = f'oneAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}'
            else: 
                method = f'mrmdAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}'
            
            if args.consecutive_time:
                entry = [f"{args.train_percent}_consec", args.ROI, method, np.mean(mse_mrmd), np.std(mse_mrmd)]
            else:
                entry = [args.train_percent, args.ROI, method, np.mean(mse_mrmd), np.std(mse_mrmd)]

            if not checkexist(outdf, dict(zip(cols[:3], entry[:3]))):
                outdf.loc[len(outdf)] = entry

    elif args.ind_mrAE:
        # individual mr-AE extension
        if args.consecutive_time:
            savepath = savepath + '_consec'
        indmrdists=[]
        for pt in range(1, args.n_pt+1):
            p_test =np.load(os.path.join(savepath,
                                         f"ind_mrAE_sub-{pt:02}_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}_testhidden.npy"))
            p_train = np.load(os.path.join(truephatepath,
                                               f"sub-{pt:02}_{embednaming}"))
            indmrdists.append(euclidean_distances(p_test, p_train))
        indmrdists = np.vstack(indmrdists).reshape(args.n_pt, len(testTRs), len(trainTRs))
        
        mse_indmr = []
        for pt in range(truedists.shape[0]):
            mse_indmr.append(mean_squared_error(truedists[pt], indmrdists[pt]))
        if args.consecutive_time:
            entry = [f"{args.train_percent}_consec", args.ROI,
                     f'ind_mrAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}',
                     np.mean(mse_indmr), np.std(mse_indmr)]
        else:
            entry = [args.train_percent, args.ROI,
                     f'ind_mrAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}',
                     np.mean(mse_indmr), np.std(mse_indmr)]

        if not checkexist(outdf, dict(zip(cols[:3], entry[:3]))):
            outdf.loc[len(outdf)] = entry

    print(outdf)

if __name__=="__main__":
    main()








