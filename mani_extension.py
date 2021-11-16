# Subset timepoints for test manifold extension within subjects
# interpolate manifold embedding on test slices of data
# MRMD-AE

import warnings  # Ignore sklearn future warning
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import argparse
import torch
import random
import phate
from lib.fMRI import fMRIAutoencoderDataset, fMRI_Time_Subjs_Embed_Dataset
from lib.helper import extract_hidden_reps, get_models, checkexist
from torch.utils.data import DataLoader
import os
from lib.utils import set_grad_req


parser = argparse.ArgumentParser()
parser.add_argument('--n_TR', type = int, default = 1976)
parser.add_argument('--train_percent', type=int, default=90)
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--ROI', type = str, default = 'early_visual')
parser.add_argument('--hidden_dim', type = int, default = 64)
parser.add_argument('--zdim', type = int, default = 20)
parser.add_argument('--n_pt', type = int, default = 16)
parser.add_argument('--volsurf', type = str, default='MNI152_3mm_data') # default is volumetric data , alternatively fsaverage_data for surface data
parser.add_argument('--input_size', type = int, default = None)
parser.add_argument('--batch_size', type = int, default =64)
parser.add_argument('--symm', action ='store_true') # use the symmetric config for encoder as decoder, so the latent encoder dim is the same as manifold dim
parser.add_argument('--lr', type = float, default=0.001)
parser.add_argument('--lam', type = float, default = 0)
parser.add_argument('--lam_mani', type = float, default = 1)
parser.add_argument('--n_epochs', type = int, default = 4000)
parser.add_argument('--shuffle_reg', action='store_true')
parser.add_argument('--ind_mrAE', action='store_true', help='set active to train independent MR-AE')
parser.add_argument('--pt', type=int, default=None)
parser.add_argument('--consecutive_time', action ='store_true', help='set active to make consecutive times e.g. 50% train will be first half of time series')
parser.add_argument('--oneAE', action='store_true', help='use a single autoencoder')
parser.add_argument('--reg_ref', action = 'store_true')

def main():
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    outfile = 'results/mrmdAE_insubject_mani_extension.csv'
    if args.ind_mrAE:
        # if independent mr-AE is used
        outfile = 'results/ind_mrAE_insubject_mani_extension.csv'
        if args.pt is None:
            print('ERROR: need to specify subject when indepent mrAE is trained, set --pt=?')
            return
        args.n_pt = 1
    if args.oneAE:
        print('using one encoder one decoder setup')
        outfile = 'results/oneAE_insubject_mani_extension.csv'

    embedpath = "/gpfs/milgram/scratch60/turk-browne/jh2752/data"
    if not os.path.exists(embedpath):
        os.makedirs(embedpath)

    path_trainTRs = f"/gpfs/milgram/scratch60/turk-browne/jh2752/data/sherlock_{args.train_percent}_trainTRs.npy"
    if args.consecutive_time:
        path_trainTRs = f"/gpfs/milgram/scratch60/turk-browne/jh2752/data/sherlock_{args.train_percent}_consec_trainTRs.npy"
    if not os.path.exists(path_trainTRs):
        if not args.consecutive_time:
            trainTRs = np.random.choice(args.n_TR, int(args.n_TR*args.train_percent/100), replace=False)
        else:
            trainTRs = np.arange(int(args.n_TR*args.train_percent/100))
        trainTRs.sort()
        np.save(path_trainTRs, trainTRs)
    else:
        trainTRs = np.load(path_trainTRs)
    testTRs = np.setxor1d(np.arange(args.n_TR), trainTRs)
    testTRs.sort()

    datapath = f"/gpfs/milgram/scratch60/turk-browne/neuromanifold/sherlock/{args.volsurf}/denoised_filtered_smoothed/ROI_data/{args.ROI}/data"
    datanaming = f"{args.ROI}_sherlock_movie.npy"
    embednaming = f"{args.ROI}_{args.zdim}dimension_{args.train_percent}_train_PHATE.npy"
    if args.consecutive_time:
        embednaming = f"{args.ROI}_{args.zdim}dimension_{args.train_percent}_consec_train_PHATE.npy"


    if not os.path.exists(os.path.join(embedpath, f"sub-01_{embednaming}")):
        print( 'prepare train embed data')
        for pt in range(1,args.n_pt+1):
            X = np.load(os.path.join(datapath, f"sub-{pt:02}_{datanaming}"))[trainTRs]
            pop = phate.PHATE(n_components = args.zdim)
            X_p = pop.fit_transform(X)
            Xtest = np.load(os.path.join(datapath, f"sub-{pt:02}_{datanaming}"))[testTRs]
            Xtest_p = pop.transform(Xtest)
            np.save(os.path.join(embedpath, f"sub-{pt:02}_{embednaming}"),
                    X_p)
            testphate_file = f"sub-{pt:02}_{embednaming}"
            testphate_file = testphate_file.replace('_train_','_test_')
            np.save(os.path.join(embedpath,testphate_file),
                    Xtest_p) # the test is phate landmark interpolation Xtest_p

    savepath = f"/gpfs/milgram/scratch60/turk-browne/tw496/results/sherlock_{args.volsurf}" \
               f"_{args.ROI}_mani_extend_{args.train_percent}"
    if args.consecutive_time:
        savepath =savepath+'_consec'
    if args.oneAE:
        savepath = savepath +'_oneAE'

    outdf = None
    cols = ['ROI', 'hidden_dim', 'zdim', 'lam_mani', 'lam_common', 'symm_design','train_percent']
    entry = [args.ROI, args.hidden_dim, args.zdim, args.lam_mani, args.lam, args.symm, args.train_percent]
    
    if args.consecutive_time:
        entry = [args.ROI, args.hidden_dim, args.zdim,
             args.lam_mani, args.lam,
             args.symm, f"{args.train_percent}_consec"]

    if args.ind_mrAE:
        print('training individual mr-AE')
        cols.append('subject')
        entry.append(args.pt)

    if os.path.exists(outfile):
        outdf_old = pd.read_csv(outfile)
        exist = checkexist(outdf_old, dict(zip(cols, entry)))
        if exist:
            print(f"{entry} exists")
            return
        else: 
            print(f"{entry} running")
    else: 
        outdf_old = None

    patient_ids = np.arange(1,args.n_pt+1)
    if args.ind_mrAE:
        patient_ids = [args.pt]
    dataset = fMRI_Time_Subjs_Embed_Dataset(patient_ids,
                                            datapath,
                                            embedpath,
                                            trainTRs,
                                            emb_name_suffix=embednaming,
                                            data_3d=False,
                                            data_name_suffix=datanaming)
    if args.input_size is None:
        args.input_size = dataset.get_TR_dims()[0]
    if args.input_size!= dataset.get_TR_dims()[0]:
        print('ERROR: input dim and args.input_size not match')
        return
    if args.zdim is None:
        args.zdim = dataset.get_embed_dims()
    if args.zdim!=dataset.get_embed_dims():
        print('ERROR: manifold layer dim and embedding reg dim not match')
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoders = get_models(args)
    encoder.to(device)

    if args.oneAE:
        decoders = [decoders[0]] # if use one encoder one decoder setup, keep only one decoder

    for i in range(len(decoders)):
        decoders[i].to(device)
    params = list(encoder.parameters())
    for decoder in decoders:
        params = params + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)  # either set initial lr or default 0.001

    criterion = torch.nn.MSELoss()  # reconstruction loss criterion
    mr_criterion = torch.nn.MSELoss()  # manifold embedding regularization criterion
    reg_criterion = torch.nn.MSELoss()

    losses = np.array([])
    rconst_losses = np.array([])
    reg_losses = np.array([])
    manifold_reg_losses = np.array([])

    pt_list = np.arange(len(patient_ids))

    for epoch in range(1, args.n_epochs + 1):
        epoch_losses = 0.0
        epoch_rconst_losses = 0.0
        epoch_manifold_reg_losses = 0.0
        epoch_reg_loss = 0.0

        for data_batch, embed_batch in dataloader:
            optimizer.zero_grad()
            current_bs = data_batch.size()[0]
            data_batch = data_batch.reshape((data_batch.shape[0]*data_batch.shape[1], -1)).float()
            data_batch = data_batch.to(device)
            embed_batch = embed_batch.reshape((embed_batch.shape[0]*embed_batch.shape[1],-1)).float()
            embed_batch = embed_batch.to(device)

            hidden = encoder(data_batch)
            hiddens = [hidden[i * current_bs:(i + 1) * current_bs] for i in range(len(patient_ids))]
            outputs = []
            embeds = []
            for i in range(len(patient_ids)):
                if args.oneAE:
                    embed, output = decoders[0](hiddens[i])
                else:
                    set_grad_req(decoders, i)
                    embed, output = decoders[i](hiddens[i])
                outputs.append(output)
                embeds.append(embed)

            if args.shuffle_reg:
                random.shuffle(pt_list)

            if args.lam >0:
                loss_reg = reg_criterion(hiddens[pt_list[0]], hiddens[pt_list[1]])
                if args.reg_ref:
                    for z1 in range(1, len(patient_ids)):
                        loss_reg += reg_criterion(hiddens[pt_list[0]], hiddens[pt_list[z1]])

                else:
                    for z1 in range(1, len(patient_ids) - 1):  # consecutive pairs (cycle)
                        z2 = z1 + 1
                        loss_reg += reg_criterion(hiddens[pt_list[z1]], hiddens[pt_list[z2]])

            loss_reconstruct = criterion(torch.stack(outputs).view(data_batch.shape), data_batch)
            loss_manifold_reg = mr_criterion(torch.stack(embeds).view(embed_batch.shape), embed_batch )

            loss =loss_reconstruct + args.lam_mani*loss_manifold_reg
            if args.lam>0:
                loss+=args.lam*loss_reg
            loss.backward()
            optimizer.step()

            epoch_losses += loss.item() * data_batch.size(0)
            epoch_rconst_losses += loss_reconstruct.item()*data_batch.size(0)
            epoch_manifold_reg_losses += loss_manifold_reg.item() * data_batch.size(0)
            if args.lam>0:
                epoch_reg_loss += loss_reg.item()*data_batch.size(0)

        epoch_losses = epoch_losses / (len(trainTRs) * len(patient_ids))  # change to report epoch loss
        epoch_rconst_losses = epoch_rconst_losses / (len(trainTRs) * len(patient_ids))
        epoch_manifold_reg_losses = epoch_manifold_reg_losses / (len(trainTRs) * len(patient_ids))
        epoch_reg_loss = epoch_reg_loss / (len(trainTRs) * len(patient_ids))

        print(f"Epoch {epoch}\tLoss={epoch_losses:.4f}\tloss_rconst={epoch_rconst_losses:.4f}\tloss_manfold_reg={epoch_manifold_reg_losses:.4f}\tloss_reg={epoch_reg_loss:.4f}")

        losses = np.append(losses, epoch_losses)
        rconst_losses = np.append(rconst_losses, epoch_rconst_losses)
        manifold_reg_losses = np.append(manifold_reg_losses, epoch_manifold_reg_losses)
        reg_losses=np.append(reg_losses, epoch_reg_loss)

    all_losses = np.stack((losses, rconst_losses, manifold_reg_losses, reg_losses), axis=1)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    lossfile = f'mrmdAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}_all_train_losses.npy'
    if args.ind_mrAE:
        lossfile = f'ind_mrAE_sub-{args.pt:02}_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}_all_train_losses.npy'
    np.save(os.path.join(savepath, lossfile), all_losses)

    modeldict = {'encoder_state_dict': encoder.state_dict()}
    for i in range(len(decoders)):
        modeldict[f"decoder_{i}_state_dict"] = decoders[i].state_dict()
    modeldict['optimizer_state_dict'] = optimizer.state_dict()
    modeldict['epoch'] = epoch

    ckptfile = f"mrmdAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}.pt"
    if args.ind_mrAE:
        ckptfile = f'ind_mrAE_sub-{args.pt:02}_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}.pt'
    torch.save(modeldict, os.path.join(savepath, ckptfile))

    # test on test TR and record the test embeddings
    dataset = fMRIAutoencoderDataset(patient_ids,
                                     datapath,
                                     testTRs,
                                     data_3d=False,
                                     data_name_suffix=datanaming)
    encoder.eval()
    hidden, al_hidden = extract_hidden_reps(encoder, decoders, dataset, device, None, args)
    hidden = hidden.reshape(args.n_pt, len(testTRs), -1)

    if args.ind_mrAE:
        hidden = hidden.reshape(len(testTRs), -1)
        hiddenfile = f"ind_mrAE_sub-{args.pt:02}_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}_testhidden.npy"
    else: 
        hiddenfile = f"mrmdAE_{args.hidden_dim}_{args.zdim}_lam{args.lam}_manilam{args.lam_mani}_symm{args.symm}_testhidden.npy"
    np.save(os.path.join(savepath, hiddenfile ), hidden)

    cols.append('hiddenfile')
    entry.append(os.path.join(savepath, hiddenfile))

    if outdf is None:
        outdf = pd.DataFrame(columns = cols)
    outdf.loc[len(outdf)]=entry

    if os.path.exists(outfile):
        outdf_old = pd.read_csv(outfile)
        outdf = pd.concat([outdf_old, outdf])
    outdf.to_csv(outfile, index=False)

if __name__=='__main__':
    main()



