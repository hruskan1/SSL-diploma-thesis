""" Hierachical VAE for CityScape
    Latent variables: Bernoulli/Categorical
    Image: RGB
    Leaning: semi-supervised
"""

import yaml
import argparse
import os
from easydict import EasyDict
import itertools
import torch
import torch.nn as nn
from svae.nets.hvae import HVAE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
from src.mixmatch.datasets import get_CityScape


def validate(mod: HVAE, loader: DataLoader, device:torch.cuda.device,ignore_class:int=0):
    """
    Function to evalute model on data in loader
    
    ignore_class:int  class which should be ignored during evaluation
    """
    cel = nn.CrossEntropyLoss(reduction='mean')
    numel = 0
    vloss = 0.0
    vacc = 0.0
    mod.eval()
    with torch.no_grad():
        for (xl, tl) in loader:
            x0_smpl = xl.to(device)
            t0_smpl = tl.to(device, dtype=torch.float)
            enc_acts = mod.encoder_activations(x0_smpl)
            scores = enc_acts[0][0]
            vloss += cel(scores, t0_smpl).mean().item()

            acc_mask = torch.argmax(scores,dim=1) == torch.argmax(t0_smpl,dim=1)
            
            # ignore targets with class "ignore_class" 
            igonre_mask = torch.argmax(t0_smpl,dim=1) == ignore_class
            acc_mask[igonre_mask]  = 0
            
            vacc += torch.sum(acc_mask)
            numel += acc_mask.numel() - igonre_mask.sum()
    
    return vloss / numel, vacc / numel


def main(args):
    # general preparations
    if torch.cuda.is_available():
        args.device = torch.device(args.device) # hot fix 
    else:
        args.device = torch.device('cpu')
    logpath = 'logs/log-' + args['basename'] + args.call_prefix + '.txt'
    logpath = os.path.join(args.res_path, logpath)
    hvae_path = 'models/' + args.basename + args.call_prefix + '.pt'
    hvae_path = os.path.join(args.res_path, hvae_path)
    viz_path = 'images/' + args.basename + args.call_prefix + '-reconstructed.png'
    viz_path = os.path.join(args.res_path, viz_path)
    stats_path = 'logs/stats-' + args.basename + args.call_prefix + '-enc.npz'
    stats_path = os.path.join(args.res_path, stats_path)

    # prepare data loaders
    
    labeled_dataloader, unlabeled_dataloader, validation_dataloader,test_dataloader =get_CityScape(root=args.dataset_path,
                                                                                                    n_labeled = args.n_labeled,
                                                                                                    n_val = args.n_val,
                                                                                                    batch_size = args.batch_size,
                                                                                                    mode = 'fine', # change if you want to
                                                                                                    size = args.img_size,
                                                                                                    target_type='semantic',
                                                                                                    verbose=True
                                                                                                    )

    print('# Starting', file=open(logpath, 'w'), flush=True)

    # model
    hvae = HVAE(**args).to(args.device)

    if args.load_prefix is not None:
        hvae_load_path = 'models/' + args.basename + args.load_prefix + '.pt'
        hvae_load_path = os.path.join(args.res_path, hvae_load_path)
        checkpoint = torch.load(hvae_load_path, map_location=args.device)
        hvae.load_state_dict(checkpoint['model_state_dict'])
        hvae.dec_optimizer.load_state_dict(checkpoint['dec_optimizer_state_dict'])
        hvae.enc_optimizer.load_state_dict(checkpoint['enc_optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
    else:
        hvae.init_weights()
        epoch = 0
    hvae.encoder_train()
    hvae.decoder_train()
    # count parameters
    dec_paraml = []
    enc_paraml = []
    for bl in hvae.sblock_list[1:]:
        dec_paraml.append(bl.decoder.parameters())
        enc_paraml.append(bl.encoder.parameters())
    dec_parameters = itertools.chain(*dec_paraml)
    enc_parameters = itertools.chain(*enc_paraml)
    pnumd = sum(p.numel() for p in dec_parameters if p.requires_grad)
    pnume = sum(p.numel() for p in enc_parameters if p.requires_grad)
    print('# parameters dec, enc: {}, {}'.format(pnumd, pnume), file=open(logpath, 'a'), flush=True)

    # learning preparation
    log_period = 1
    save_period = 10
    niterations = args.niterations
    count = 0
    start_acc = True
    acc_dt = 0.0
    acc_et = 0.0

    labeled_train_iter = iter(labeled_dataloader)
    unlabeled_train_iter = iter(unlabeled_dataloader)

    #print(f"{hvae.sblock_list=}")


    while True:
        # train epoch
        x0_smpl = None

        
        k_imgs_seen = 0
        while k_imgs_seen < args.kimg:
            # Iterate over the end if necessary (Can be used with different sizes of dataloaders)
            try:
                xl, tl = next(labeled_train_iter)
            except:
                labeled_train_iter = iter(labeled_dataloader)
                xl, tl = next(labeled_train_iter)

            try:
                xu,_ = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_dataloader)
                xu,_ = next(unlabeled_train_iter)

            current_batch_size = min(xl.shape[0],xu.shape[0])
        
            xl = xl[:current_batch_size]
            xu = xu[:current_batch_size]
            tl = tl[:current_batch_size]

            k_imgs_seen += current_batch_size
        
            # supervised batch 
            xl  = xl.to(args.device)
            tl  = tl.to(args.device, dtype=torch.float) # already one-hot
            

            # =====  decoder learn step (supervised learning) ====
            
            with torch.no_grad():
                enc_acts = hvae.encoder_activations(xl)
                act = enc_acts[0]
                zl0_smpl = hvae.sblock_list[0].sample(act)
                # z0[0] is segmentation, z0[1] is bernulli 
                zl0_smpl[0] = tl
            dt = hvae.decoder_learn_step(xl, z0=zl0_smpl)
            
            # =====  encoder learn step (supervised learning) ====
            z0_shape = hvae.z0_shape
            
            for _ in range(1):
                z0x = hvae.z0_prior_sample(z0_shape)
                z0x[0] = tl
                et = hvae.encoder_learn_step(z0x)

            # print(f"{hvae.z0_shape=}")
            # print(f"{len(z0_smpl)=},{z0_smpl[0].shape},{z0_smpl[1].shape}")

            xu = xu.to(args.device)
            # ====== decoder learn step (unsupervised) ======
            dt = hvae.decoder_learn_step(xu)

            # =====  encoder learn step (unsupervised) ======
            z0_shape = hvae.z0_shape
            for _ in range(1):
                zu0_smpl = hvae.z0_prior_sample(z0_shape)
                et = hvae.encoder_learn_step(zu0_smpl)

            # accumulate data terms
            if start_acc:
                acc_dt = dt.item()
                acc_et = et.item()
                start_acc = False

            acc_dt = acc_dt * 0.999 + dt.item() * 0.001
            acc_et = acc_et * 0.999 + et.item() * 0.001

            print(f"\r{count=}|({k_imgs_seen}/{args.kimg})",end='')

        if (count % log_period == log_period-1) or (count == niterations-1):
            jsdivs = hvae.sblock_list[0].jsdivs
            print(f'jsdivs: {len(jsdivs)=},{jsdivs[0].shape},{jsdivs[1].shape}')
            cat_jsdivs =  jsdivs[0].mean().item()
            bin_jsdivs =  jsdivs[1].mean().item()
            strtoprint = 'epoch: '+ str(count + epoch)
            strtoprint += ' dt: {:.4}'.format(-acc_dt)
            strtoprint += ' et: {:.4}'.format(-acc_et)
            strtoprint += ' cat_jsdivs: {:.4}'.format(cat_jsdivs)
            strtoprint += ' bin_jsdivs: {:.4}'.format(bin_jsdivs)
            
            val_loss, val_acc = validate(hvae, test_dataloader, args.device)
            strtoprint += ' vloss: {:.4}'.format(val_loss)
            strtoprint += ' vacc: {:.4}'.format(val_acc)

            print(strtoprint, file=open(logpath, 'a'), flush=True)


        if (count % save_period == save_period-1) or (count == niterations-1):
            print('# Saving models ...', file=open(logpath, 'a'), flush=True)
            checkpoint = {
                'model_state_dict': hvae.state_dict(),
                'enc_optimizer_state_dict': hvae.enc_optimizer.state_dict(),
                'dec_optimizer_state_dict': hvae.dec_optimizer.state_dict(),
                'epoch': (count + epoch)}
            torch.save(checkpoint, hvae_path)
            # save encoder stats
            hvae.save_stats(stats_path)

            # save a few reconstructions
            with torch.no_grad():
                x0 = xl[0:16]
                z0x = z0x[0:16]
                u0 = xu[0:16]
                z0u = zu0_smpl[0:16]
                t0 = tl[0:16]
                # reconstruction
                _, probs = hvae.posterior_sample(x0)
                x1 = probs[-1]
                # sample image from latent
                (x3, x2) = hvae.prior_sample(z0x)
                # continue sampling (limiting marginal distrib)
                
                # supervised 
                for _ in range(100):
                    (z_all, _) = hvae.posterior_sample(x3, expanded=True)
                    z_all[0][0] = t0
                    (x3, probs) = hvae.posterior_sample(x3, z0=z_all[0])
                x3 = probs[-1]
                # unsupervised

                # reconstruction
                _, probs = hvae.posterior_sample(u0)
                u1 = probs[-1]
                (u3,u2) = hvae.prior_sample(z0u)
                for _ in range(100):
                    (u3, _) = hvae.posterior_sample(u3)
                (_, probs) = hvae.posterior_sample(u3)

                u3 = probs[-1]
                xvis = torch.cat((x0, x1, x2, x3,u0,u1,u2,u3))
                xvis.clamp_(0.0, 1.0)

                #TODO: Take care about correct color visualization
                
                vutils.save_image(xvis, viz_path, normalize=True, nrow=8)

        count += 1
        if count == niterations:
            break


if __name__ == '__main__':
    # read parameters from config file
    dscr = 'Bernoulli/Categorical hierarchical VAE for CityScape dataset with symmetric learning'
    parser = argparse.ArgumentParser(description=dscr)
    parser.add_argument('-c', '--config',
                        help='name of the config file (yaml)')
    cmdl_args = parser.parse_args()
    if cmdl_args.config:
        cfg_name = cmdl_args.config
    else:
        cfg_name = 'shvae-config-unet-rescale.yaml'
    with open(cfg_name, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    ed = EasyDict(cfg)
    main(ed)
