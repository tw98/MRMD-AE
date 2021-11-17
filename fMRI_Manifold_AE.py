###########################################################################
## latent manifold embedding regularized autoencoder for fMRI data
## one encoder, multiple decoders
###########################################################################

import warnings  # Ignore sklearn future warning
warnings.simplefilter(action='ignore', category=FutureWarning)
from lib.utils import setup_save_n_log, set_grad_req, setup_save_dir
import numpy as np
import torch
import logging
from lib.fMRI import fMRI_Time_Subjs_Embed_Dataset
from torch.utils.data import DataLoader
from lib.autoencoder import Encoder_basic, Decoder_basic, Decoder_Manifold
import random
from lib.utils import save_checkpoint_allin1
import os
import argparse
from lib.helper import get_models, plot_losses
import pandas as pd
from fMRI_Manifold_downstream import downstream_analysis

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 0)

parser.add_argument('--experiment_folder', type=str, default='.')
parser.add_argument('--n_subjects', type=int, default=16) # number of subjects
parser.add_argument('--datapath', type = str, default='./data/ROI_data/early_visual/fMRI')
parser.add_argument('--datanaming', type=str, default='early_visual_sherlock_movie.npy') #file name base, s.t. filename = sub-??_datanaming
parser.add_argument('--embedpath',type=str, default='./data/ROI_data/early_visual/embedding')
parser.add_argument('--embednaming', type=str, default='')
parser.add_argument('--summary_file', type=str, default='./results/fMRI_manifold_AE_summary.csv')

parser.add_argument('--n_timerange', type = int, default=1976) # number of timepoints in dataset
parser.add_argument('--train_half', type = int, default=None) # train_half = 1 is first half, 2 is second half
parser.add_argument('--hidden_dim', type = int, default=64) # dimension of common embedding layer
parser.add_argument('--zdim', type = int, default=20) # dimension of bottleneck manifold layer

parser.add_argument('--batch_size', type = int, default=64) 
parser.add_argument('--n_epochs', type = int, default=4000)
parser.add_argument('--lr', type = float, default=0.001)
parser.add_argument('--lam', type = float, default=10) # common embedding layer regularization parameter
parser.add_argument('--lam_mani', type = float, default=1) # individual manifold layerregularization parameter
parser.add_argument('--lam_decay', type = float, default=1) # regularization decay factor

parser.add_argument('--save_model', action='store_true')
parser.add_argument('--save_freq', type = int, default = 1)
parser.add_argument('--loadpath', type =str, default=None) #directory that saves checkpoints
parser.add_argument('--load_epoch', type = int, default=0)

parser.add_argument('--shuffle_reg', action='store_true')
parser.add_argument('--reg_ref', action = 'store_true') # for regularization, use pt[0] (shuffle or not) as reference; if not activated, random nonoverlapping pairs cycle
parser.add_argument('--labelpath', type = str, default='./data/sherlock_labels_coded_expanded.csv')
parser.add_argument('--xsubj', action='store_true', help='train for cross subject, decode all subject input using all subjects decoders')
parser.add_argument('--lam_xsubj', type = float, default=1)
parser.add_argument('--downstream', action = 'store_true') #follow with downstream analyses
parser.add_argument('--input_size', type = int, default=0)

class ExperimentParameters():
    def __init__(self, args) -> None:
        self.roi = self.set_ROI(args)
        self.manifoldtype = self.set_manifold(args)
        self.data_3d = False

        self.patient_ids = self.set_patient_ids(args)

        self.resultsdf = None

        self.savepath = self.set_save_path(args)

        self.set_checkpoint_path(args)

    def set_ROI(self, args):
        ROIs = [
            'aud_early','early_visual_neurosynth', 'early_visual',
            'pmc_nn','auditory_neurosynth', 'fusiform_face_neurosynth', 
            'late_visual_neurosynth', 'objects_neurosynth'
            ]

        for ROIi in ROIs:
            if ROIi in args.datapath:
                return ROIi

    def set_manifold(self, args):
        if '_TPHATE' in args.embednaming:
            return 'TPHATE'
        elif '_PHATE' in args.embednaming:
            return 'PHATE'

    def set_patient_ids(self, args):
        return np.arange(1,args.n_subjects+1)

    def set_results_df(self, args):
        self.resultsdf = pd.read_csv(args.summary_file)
        self.resultsdf = self.resultsdf.loc[(self.resultsdf.ROI==self.roi) & (self.resultsdf['manifold_type']==self.manifoldtype) &
                                  (self.resultsdf['common_hiddendim']==args.hidden_dim) & (self.resultsdf['manifold_embed_dim']==args.zdim) &
                                  (self.resultsdf['lambda_mani']==args.lam_mani) & (self.resultsdf['lambda_common']==args.lam) &
                                  (self.resultsdf['train_half']==args.train_half)
                                  ]
        if args.xsubj:
            self.resultsdf = self.resultsdf.loc[self.resultsdf.lam_trans==args.lam_xsubj]

        if self.resultsdf.shape[0]>0:
            print(f"The combination has run: {self.roi}+{self.manifoldtype}+commonhid={args.hidden_dim}+zdim={args.zdim}+commonlam={args.lam}+manilam={args.lam_mani}",
                  f"train_half={args.train_half}")
            exit()
        else:
            print(f"Running combination: {self.roi}+{self.manifoldtype}+commonhid={args.hidden_dim}+zdim={args.zdim}+commonlam={args.lam}+manilam={args.lam_mani}",
                  f"train_half={args.train_half}")

    def set_save_path(self, args):
        savepath = f'results/sherlock_{self.roi}_{args.n_subjects}pt_hiddendim{args.hidden_dim}_bs{args.batch_size}_' \
                f'reg_lam{args.lam}_manilam{args.lam_mani}'

        embedsource = args.embedpath.split('/')
        SRMstate = ''
        for entry in embedsource:
            if 'SRM' in entry:
                SRMstate=entry

        if '_PHATE' in args.embednaming:
            savepath+=f'_{SRMstate}_phate{args.zdim}'
        if '_TPHATE' in args.embednaming:
            savepath +=f'_{SRMstate}_tphate{args.zdim}'
        if args.shuffle_reg:
            savepath+='_shf'

        print(f"save to {savepath}")

        if args.train_half is not None:
            if args.train_half ==1:
                savepath+=f'_{1}half'
            else:
                savepath += f'_{2}half'
        

        savepath = setup_save_n_log(savepath)
        print('set up log save to %s' % savepath)
        
        return savepath

    def set_checkpoint_path(self, args):
        self.chkpt_savepath = os.path.join(args.experiment_folder, self.savepath)
        self.chkpt_savepath = setup_save_dir(self.chkpt_savepath)
        logging.info(f'checkpoints save to: {self.chkpt_savepath}')


def main():
    # params
    args = parser.parse_args()

    param = ExperimentParameters(args)

    if os.path.exists(args.summary_file):
        param.set_results_df()  

    # set up train_half for dataset
    if args.train_half is not None:
        args.n_timerange = args.n_timerange // 2
        if args.train_half == 1:
            train_half = np.arange(args.n_timerange)
        else:
            train_half = np.arange(args.n_timerange, 2*args.n_timerange)
    else:
        train_half = np.arange(args.n_timerange)

    # make training reproducible
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    if args.train_half==1:
        logging.info(f"train on first half data with embedding ending with {args.embednaming}")
        if 'split1' in args.embednaming:
            logging.info('ERROR: train on first half need split0 embedding file!')

    elif args.train_half==2:
        logging.info(f"train on second half data with embedding ending with {args.embednaming}")
        if 'split0' in args.embednaming:
            logging.info('ERROR: train on second half need split1 embedding file!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('Device:%s' % device)
    logging.info(f"AE model, manifold regularized")
    logging.info(f"Network params: hidden dim={args.hidden_dim}, manifold embedding dim={args.zdim}")
    logging.info(f"batch size={args.batch_size}, lr={args.lr}, seed={args.seed}")
    logging.info(f'save checkpoint: {args.save_model}')

    if args.lam==0:
        logging.info(f'common embedding regularization not used.')
    if args.train_half is not None:
        logging.info(f'TR: {train_half[0]}-{train_half[-1]}')

    embed_name_suffix = args.embednaming
    dataset = fMRI_Time_Subjs_Embed_Dataset(param.patient_ids,
                                            args.datapath,
                                            args.embedpath,
                                            train_half,
                                            emb_name_suffix=embed_name_suffix,
                                            data_3d=param.data_3d,
                                            data_name_suffix = args.datanaming)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    args.input_size = dataset.get_TR_dims()[0]
    embed_size = dataset.get_embed_dims()
    args.zdim = embed_size # this is to set the manifold embedding shape

    logging.info(f"input size={args.input_size}")
    logging.info(f"manifold embedding size={args.zdim}")

    encoder, decoders = get_models(args)
    encoder.to(device)
    for i in range(len(decoders)):
        decoders[i].to(device)

    # initialize optimizer
    params = list(encoder.parameters())
    for decoder in decoders:
        params = params + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)  # either set initial lr or default 0.001

    # load pretrained and continue
    if args.loadpath is not None:
        print(f'loading model from {args.loadpath} at checkpoint epoch {args.load_epoch}')
        checkpoint = torch.load(os.path.join(args.loadpath, f'ae_e{args.load_epoch}.pt'))
        encoder.load_state_dict(checkpoint['encoder_state_dict'])

        for i in range(1, args.n_subjects + 1):
            decoders[i - 1].load_state_dict(checkpoint[f'decoder_{i}_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info('loaded pretrained model for continue training')

    criterion = torch.nn.MSELoss()  # reconstruction loss criterion
    mr_criterion = torch.nn.MSELoss() # manifold embedding regularization criterion

    # common latent regularization
    reg_criterion = torch.nn.MSELoss()

    losses = np.array([])
    rconst_losses = np.array([])
    reg_losses = np.array([])
    manifold_reg_losses = np.array([])

    pt_list = np.arange(args.n_subjects)

    for epoch in range(args.load_epoch + 1, args.n_epochs + 1):
        epoch_losses = 0.0
        epoch_rconst_losses = 0.0
        epoch_manifold_reg_losses = 0.0
        epoch_reg_losses=0.0
        epoch_trans_losses = 0.0

        for data_batch, embed_batch in dataloader:
            optimizer.zero_grad()
            current_bs = data_batch.size()[0]

            data_batch = data_batch.reshape((data_batch.shape[0]*data_batch.shape[1], -1)).float()
            data_batch = data_batch.to(device)

            embed_batch = embed_batch.reshape((embed_batch.shape[0]*embed_batch.shape[1],-1)).float()
            embed_batch = embed_batch.to(device)

            hidden = encoder(data_batch)
            hiddens = [hidden[i * current_bs:(i + 1) * current_bs] for i in range(args.n_subjects)]

            if args.xsubj:
                outputs = []
                embeds = []
                for i in range(args.n_subjects):
                    set_grad_req(decoders, i)
                    embed, output = decoders[i](hidden) # here the embed and output are of [T_batch x n_subjects, *] shape
                    outputs.append(output.reshape((len(pt_list), -1, args.input_size)))
                    embeds.append(embed.reshape((len(pt_list), -1, args.zdim)))
                outputs = torch.stack(outputs)
                embeds = torch.stack(embeds)

            else:
                outputs = []
                embeds = []
                for i in range(args.n_subjects):
                    set_grad_req(decoders, i)
                    embed, output = decoders[i](hiddens[i])
                    outputs.append(output)
                    embeds.append(embed)

            if args.shuffle_reg:
                random.shuffle(pt_list)
            loss_reg = reg_criterion(hiddens[pt_list[0]], hiddens[pt_list[1]])

            if args.reg_ref:
                for z1 in range(1, args.n_subjects):
                    loss_reg += reg_criterion(hiddens[pt_list[0]], hiddens[pt_list[z1]])
            else:
                for z1 in range(1, args.n_subjects - 1):  # consecutive pairs (cycle)
                    z2 = z1 + 1
                    loss_reg += reg_criterion(hiddens[pt_list[z1]], hiddens[pt_list[z2]])

            if args.xsubj:
                reconstruct = []
                for i in range(len(param.patient_ids)):
                    reconstruct.append(outputs[i, i, :, :])
                loss_reconstruct = criterion(torch.stack(reconstruct).view(data_batch.shape), data_batch)
                translate = []
                for i in range(len(param.patient_ids)):
                    translate.append(outputs[i,
                                     np.setxor1d(np.arange(outputs.shape[0]), [i]), :, :])
                translate = torch.stack(translate)  # shape is 16 x 15 x T_bach , *
                loss_translate = criterion(translate[:, 0, :, :].reshape(data_batch.shape), data_batch)
                for i in range(1, translate.shape[1]):
                    loss_translate = loss_translate + criterion(translate[:, i, :, :].reshape(data_batch.shape),
                                                                data_batch)

                loss_manifold_reg = mr_criterion(embeds[:, 0, :, :].reshape(embed_batch.shape), embed_batch)
                for i in range(1, len(param.patient_ids)):
                    loss_manifold_reg = loss_manifold_reg + mr_criterion(embeds[:, i, :, :].reshape(embed_batch.shape),
                                                                         embed_batch)
                loss = loss_reconstruct + args.lam_xsubj * loss_translate

            else:
                loss_reconstruct = criterion(torch.stack(outputs).view(data_batch.shape), data_batch)
                loss_manifold_reg = mr_criterion(torch.stack(embeds).view(embed_batch.shape), embed_batch )
                loss_translate=0
                loss =loss_reconstruct

            if args.lam_mani>0:
                loss = loss + args.lam_mani * loss_manifold_reg
            if args.lam>0:
                loss+=args.lam*loss_reg

            args.lam_mani = args.lam_mani * args.lam_decay
            args.lam = args.lam * args.lam_decay

            loss.backward()
            optimizer.step()

            epoch_losses += loss.item() * data_batch.size(0)
            epoch_rconst_losses += loss_reconstruct.item()*data_batch.size(0)
            epoch_reg_losses +=loss_reg.item() * data_batch.size(0)
            epoch_manifold_reg_losses += loss_manifold_reg.item() * data_batch.size(0)
            if args.xsubj:
                epoch_trans_losses += loss_translate.item() * data_batch.size(0) / (args.n_subjects - 1)

        epoch_losses = epoch_losses/ (args.n_subjects * args.n_timerange)  # change to report epoch loss
        epoch_rconst_losses = epoch_rconst_losses / (args.n_subjects * args.n_timerange)
        epoch_reg_losses = epoch_reg_losses / (args.n_subjects * args.n_timerange)
        epoch_manifold_reg_losses = epoch_manifold_reg_losses/(args.n_subjects * args.n_timerange)

        logging.info(
            f"Epoch {epoch}\tLoss={epoch_losses:.4f}\tloss_rconst={epoch_rconst_losses:.4f}\ttranslate={epoch_trans_losses:.4f}\tloss_manfold_reg={epoch_manifold_reg_losses:.4f}\tloss_reg={epoch_reg_losses:.4f}")

        losses = np.append(losses, epoch_losses)
        rconst_losses = np.append(rconst_losses, epoch_rconst_losses)
        reg_losses = np.append(reg_losses, epoch_reg_losses)
        manifold_reg_losses = np.append(manifold_reg_losses, epoch_manifold_reg_losses)

        if (epoch % args.save_freq == 0 or epoch == args.n_epochs) and args.save_model:
            model_namelist = ['encoder'] + ['decoder_%d' % i for i in range(1, len(decoders) + 1)]
            save_checkpoint_allin1([encoder] + decoders, model_namelist,
                                   optimizer,
                                   os.path.join(param.chkpt_savepath, 'ae_e%d.pt' % epoch),
                                   epoch)
            logging.info(f'saved checkpoint at epoch{epoch}')
    
    
    all_losses = np.stack((losses, rconst_losses, manifold_reg_losses, reg_losses), axis=1)
    np.save(os.path.join(param.savepath, 'all_train_losses.npy'), all_losses)

    # plot_losses(args, all_losses, len(dataloader), len(dataset.TRs), param.savepath)

    del dataloader
    del dataset

    if args.downstream:
        downstream_analysis(args, param, device, encoder, decoders)

        
if __name__ == '__main__':
    main()