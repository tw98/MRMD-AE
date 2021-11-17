##########################################################
## MDMR-AE: extend to new subject
## train MDMR-AE on 15 subjects and test on the other subject
## only the trained common encoder can be applied on the other subject
## with the trained common encoder fixed, also train a generic shared decoder to get the bottleneck layer
## if without a generic shared decoder, translate to each of the train subjects and get average
###########################################################

import warnings  # Ignore sklearn future warning
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch
from lib.fMRI import fMRIAutoencoderDataset, fMRI_Time_Subjs_Embed_Dataset
from torch.utils.data import DataLoader
from lib.autoencoder import Encoder_basic, Decoder_Manifold
import random
import os
import argparse
from lib.helper import extract_hidden_reps
import pandas as pd
from lib.utils import set_grad_req

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--n_timepoints', type = int, default=1976)
parser.add_argument('--n_subjects', type = int, default=16)
parser.add_argument('--testpt', type = int, default = 16)
parser.add_argument('--train_half', type = int, default=None) #train_half=1 is first half, 2 is second
parser.add_argument('--embednaming', type=str, default=None)
parser.add_argument('--ROI', type = str, default = 'early_visual')
parser.add_argument('--hidden_dim', type = int, default = 64)
parser.add_argument('--zdim', type = int, default =20)
parser.add_argument('--batch_size', type = int, default=64)
parser.add_argument('--lr', type = float, default=0.001)
parser.add_argument('--n_epochs', type = int, default=4000)
parser.add_argument('--lam', type = float, default = 10) # for this x-subject application set to zero
parser.add_argument('--lam_xsubj', type = float, default=0.01)
parser.add_argument('--lam_mani', type = float, default=1)
parser.add_argument('--lam_decay', type = float, default=1)
parser.add_argument('--train_decoder_generic', action='store_true')  # if used, train a generic decoder in parallel
parser.add_argument('--shuffle_reg', action='store_true')
parser.add_argument('--reg_ref', action = 'store_true')
parser.add_argument('--xsubj', action='store_true', help='xsubj loss, train for cross subject, decode all subject input using all subjects decoders')

def main():
    args = parser.parse_args()

    datapath = f"./data/ROI_data/{args.ROI}/fMRI"
    embedpath = f'./data/ROI_data/{args.ROI}/embedding'
    embed_name_suffix = args.embednaming

    savepath = f"./results/sherlock_{args.ROI}_xsubj_mdmAE_hiddim{args.hidden_dim}_zdim{args.zdim}_lam{args.lam}_lammani_{args.lam_mani}"
    
    if args.xsubj:
        savepath =savepath+f'_xsubjloss{args.xsubj}_translam{args.lam_xsubj}'
    if args.train_half ==1:
            savepath+=f'_{1}half'
    else:
        savepath += f'_{2}half'
    datanaming = f"{args.ROI}_sherlock_movie.npy"
    data_3d = False

    patient_ids = np.setxor1d(np.arange(1, args.n_subjects+1), [args.testpt]) # use all but one as train subjects
    testpt = [args.testpt] # the test subject
    
    if args.train_half is not None:
        args.n_timepoints = args.n_timepoints // 2
        if args.train_half ==1:
            TR_range = np.arange(args.n_timepoints)
        else:
            TR_range = np.arange(args.n_timepoints, 2*args.n_timepoints)
    else:
        TR_range = np.arange(args.n_timepoints)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # load data of training subjects
    dataset = fMRI_Time_Subjs_Embed_Dataset(patient_ids,
                                            datapath,
                                            embedpath,
                                            TR_range,
                                            emb_name_suffix=embed_name_suffix,
                                            data_3d=data_3d,
                                            data_name_suffix=datanaming)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    input_size = dataset.get_TR_dims()[0]
    
    embed_size = dataset.get_embed_dims()
    
    if args.zdim is None:
        args.zdim =embed_size
    if args.zdim != embed_size:
        print('ERROR: manifold layer dim and embedding reg dim not match')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Encoder_basic(input_size, args.hidden_dim * 4, args.hidden_dim * 2, args.hidden_dim)
    encoder.to(device)

    decoders = []
    for i in range(len(patient_ids)):
        decoder = Decoder_Manifold(embed_size, args.hidden_dim, args.hidden_dim * 2, args.hidden_dim * 4,
                                    input_size)
        decoder.to(device)
        decoders.append(decoder)

    # add a generic decoder that is not regularized
    decoder_gen = Decoder_Manifold(embed_size, args.hidden_dim, args.hidden_dim * 2, args.hidden_dim * 4, input_size)
    decoder_gen.to(device)
    optimizer_dg = torch.optim.Adam(decoder_gen.parameters())

    params = list(encoder.parameters())
    for decoder in decoders:
        params = params + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)  # either set initial lr or default 0.001

    criterion = torch.nn.MSELoss()  # reconstruction loss criterion
    mr_criterion = torch.nn.MSELoss()  # manifold embedding regularization criterion

    # common latent regularization
    reg_criterion = torch.nn.MSELoss()

    losses = np.array([])
    rconst_losses = np.array([])
    reg_losses = np.array([])
    manifold_reg_losses = np.array([])
    rconst_generic_decoder_losses = np.array([])

    pt_list= np.arange(len(patient_ids)) # one pt is left out for testing and we have 15 pts for training

    for epoch in range(1, args.n_epochs + 1):
        epoch_losses = 0.0
        epoch_rconst_losses = 0.0
        epoch_manifold_reg_losses = 0.0
        epoch_generic_decoder_loss = 0.0
        epoch_reg_losses = 0.0
        epoch_trans_losses = 0.0

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

            if args.xsubj:
                for i in range(len(patient_ids)):
                    set_grad_req(decoders, i)
                    embed, output = decoders[i](hidden) # here the embed and output are of [T_batch x n_subjects, *] shape
                    outputs.append(output.reshape((len(pt_list), -1, args.input_size)))
                    embeds.append(embed.reshape((len(pt_list), -1, args.zdim)))
                outputs = torch.stack(outputs)
                embeds = torch.stack(embeds)
            else:
                for i in range(len(patient_ids)):
                    set_grad_req(decoders, i)
                    embed, output = decoders[i](hiddens[i])
                    outputs.append(output)
                    embeds.append(embed)

            if args.shuffle_reg:
                random.shuffle(pt_list)
            
            loss_reg = reg_criterion(hiddens[pt_list[0]], hiddens[pt_list[1]])
            
            if args.reg_ref:
                for z1 in range(1, len(patient_ids)):
                    loss_reg += reg_criterion(hiddens[pt_list[0]], hiddens[pt_list[z1]])
            else:
                for z1 in range(1, len(patient_ids) - 1):  # consecutive pairs (cycle)
                    z2 = z1 + 1
                    loss_reg += reg_criterion(hiddens[pt_list[z1]], hiddens[pt_list[z2]])
            
            if args.xsubj:
                reconstruct = []
                for i in range(len(patient_ids)):
                    reconstruct.append(outputs[i,i,:,:])
                loss_reconstruct = criterion(torch.stack(reconstruct).view(data_batch.shape), data_batch)
                translate = []
                for i in range(len(patient_ids)):
                    translate.append(outputs[i,
                                     np.setxor1d(np.arange(outputs.shape[0]),[i]),:,:])
                translate = torch.stack(translate) # shape is 15 x 14 x T_bach , *
                loss_translate=criterion(translate[:,0,:,:].reshape(data_batch.shape), data_batch)
                for i in range(1, translate.shape[1]):
                    loss_translate = loss_translate+criterion(translate[:,i,:,:].reshape(data_batch.shape), data_batch)

                loss_manifold_reg = mr_criterion(embeds[:,0,:,:].reshape(embed_batch.shape), embed_batch)
                for i in range(1, len(patient_ids)):
                    loss_manifold_reg = loss_manifold_reg+mr_criterion(embeds[:,i,:,:].reshape(embed_batch.shape), embed_batch)
                loss = loss_reconstruct + args.lam_xsubj * loss_translate
            else:
                loss_reconstruct = criterion(torch.stack(outputs).view(data_batch.shape), data_batch)
                loss_manifold_reg = mr_criterion(torch.stack(embeds).view(embed_batch.shape), embed_batch )
                loss_translate = 0
                loss = loss_reconstruct

            if args.lam_mani > 0:
                loss =loss + args.lam_mani*loss_manifold_reg
            
            if args.lam >0:
                loss = loss+ args.lam * loss_reg
            loss.backward()
            optimizer.step()

            if args.train_decoder_generic:
                embed, output = decoder_gen(hidden.detach())
                outputs = [output[i * current_bs: (i + 1) * current_bs] for i in range(len(patient_ids))]
                loss_gen_rconst = criterion(torch.stack(outputs).view(data_batch.shape), data_batch)
                loss_gen_rconst.backward()
                optimizer_dg.step()
                epoch_generic_decoder_loss+=loss_gen_rconst.item()*data_batch.size(0)

            epoch_losses += loss.item() * data_batch.size(0)
            epoch_rconst_losses += loss_reconstruct.item()*data_batch.size(0)
            epoch_manifold_reg_losses += loss_manifold_reg.item() * data_batch.size(0)
            epoch_reg_losses += loss_reg.item() * data_batch.size(0)
            
            if args.xsubj:
                epoch_trans_losses+=loss_translate.item() * data_batch.size(0)/(args.n_subjects-1)

        epoch_losses = epoch_losses / (args.n_timepoints * len(patient_ids))  # change to report epoch loss
        epoch_rconst_losses = epoch_rconst_losses / (args.n_timepoints * len(patient_ids))
        epoch_reg_losses = epoch_reg_losses / (args.n_timepoints * len(patient_ids))
        epoch_manifold_reg_losses = epoch_manifold_reg_losses / (args.n_timepoints * len(patient_ids))
        epoch_trans_losses = epoch_trans_losses/ (args.n_timepoints * len(patient_ids))

        print(
            f"Epoch {epoch}\tLoss={epoch_losses:.4f}\tloss_rconst={epoch_rconst_losses:.4f}\t"
            f"translate={epoch_trans_losses:.4f}\tloss_manfold_reg={epoch_manifold_reg_losses:.4f}\tloss_reg={epoch_reg_losses:.4f}")

        if args.train_decoder_generic:
            epoch_generic_decoder_loss = epoch_generic_decoder_loss/(args.n_timepoints * len(patient_ids))
            rconst_generic_decoder_losses = np.append(rconst_generic_decoder_losses,epoch_generic_decoder_loss)

        losses = np.append(losses, epoch_losses)
        rconst_losses = np.append(rconst_losses, epoch_rconst_losses)
        manifold_reg_losses = np.append(manifold_reg_losses, epoch_manifold_reg_losses)

    if args.train_decoder_generic:
        all_losses = np.stack((losses, rconst_losses, manifold_reg_losses, rconst_generic_decoder_losses), axis=1)
    else:
        all_losses = np.stack((losses, rconst_losses, manifold_reg_losses), axis=1)

    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    np.save(os.path.join(savepath, f'testpt{args.testpt}_all_train_losses.npy'), all_losses)

    # save encoder and decoders
    modeldict = {'encoder_state_dict': encoder.state_dict()}
    for i in range(len(decoders)):
        modeldict[f"decoder_{i}_state_dict"] = decoders[i].state_dict()
    modeldict['optimizer_state_dict'] = optimizer.state_dict()
    modeldict['epoch'] = epoch

    torch.save(modeldict, os.path.join(savepath, f"all_but_pt{args.testpt}.pt"))

    if args.train_decoder_generic:
        torch.save({
            'model_state_dict': decoder_gen.state_dict(),
            'optimizer_state_dict': optimizer_dg.state_dict(),
            'epoch': epoch
        }, os.path.join(savepath, f"all_but_pt{args.testpt}_generic_decoder.pt"))

    # load test subject's data
    dataset = fMRIAutoencoderDataset([args.testpt],
                                     datapath,
                                     TR_range,
                                     data_3d=data_3d,
                                     data_name_suffix=datanaming)
    encoder.eval()
    if args.train_decoder_generic:
        hidden, alhidden = extract_hidden_reps(encoder, [decoder_gen], dataset, device, None, args)
    else:
        # if we didn't have a generic decoder just use any of the trained decoders
        hidden= []
        for decoder in decoders:
            hidden_i, alhidden = extract_hidden_reps(encoder, [decoder], dataset, device, None, args)  
            hidden.append(hidden_i)
        hidden = np.vstack(hidden)

    hidden = hidden.reshape(-1, args.n_timepoints, args.zdim)
    alhidden = alhidden.reshape(1, args.n_timepoints, -1)

    # save test subject's represention in common embedding layer
    saveto = os.path.join(savepath, f"e{args.n_epochs}_testsubj{args.testpt}_amlp.npy")
    np.save(saveto, alhidden)

    # save test subject's represention in manifold layer
    saveto = os.path.join(savepath, f"e{args.n_epochs}_testsubj{args.testpt}_manihidden.npy")
    np.save(saveto, hidden)


if __name__ =='__main__':
    main()