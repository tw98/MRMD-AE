import datetime
import os
import logging
import torch


def setup_save_dir(savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    else:
    #     if the target directory exists, we would avoid overwriting it
    #     make using the _i
        i = 1
        while os.path.exists(savepath+'_%d'%i):
            i+=1
        savepath = savepath +'_%d'%i
        os.makedirs(savepath)
    return savepath

def setup_save_n_log(savepath):
    # check if the savepath exists
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    else:
    #     if the target directory exists, we would avoid overwriting it
    #     make using the _i
        i = 1
        while os.path.exists(savepath+'_%d'%i):
            i+=1
        savepath = savepath +'_%d'%i
        os.makedirs(savepath)

    log_fn = os.path.join(savepath, "LOG.{0}.txt".format(datetime.date.today().strftime("%y%m%d")))
    logging.basicConfig(filename=log_fn, filemode='w', level=logging.INFO)

    return savepath # we may have changed the savepath.


def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)


def save_checkpoint_allin1(model_list, model_name_list, optimizer, save_path, epoch):
    model_sd_idx = [name+'_state_dict' for name in model_name_list]
    tosave = dict(zip(model_sd_idx, [model.state_dict() for model in model_list]))
    tosave['optimizer_state_dict']= optimizer.state_dict()
    tosave['epoch'] = epoch
    torch.save(tosave, save_path)


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch


def get_per_pt(many_pts_shape, num_pts):
    per_pt_shape = list(many_pts_shape) # Have (BS * num_pts, hidden_dim)
    per_pt_shape[0] /= num_pts # Get (BS, hidden_dim)
    
    return [int(x) for x in per_pt_shape]


def set_grad_req(model_list, idx):
    for param in model_list[idx].parameters():
        param.requires_grad = True
