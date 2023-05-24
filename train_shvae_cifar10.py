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
import torch.nn.functional as F
from svae.nets.hvae import HVAE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar 

def validate(mod: HVAE, loader: DataLoader, device:torch.cuda.device):
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
            t0_smpl = nn.functional.one_hot(tl, num_classes=10).to(device, dtype=torch.float)
            t0_smpl = t0_smpl.reshape(*t0_smpl.shape,1,1)
            enc_acts = mod.encoder_activations(x0_smpl)
            scores = enc_acts[0][0]
            vloss += cel(scores, t0_smpl).sum().item()

            acc_mask = torch.argmax(scores,dim=1) == torch.argmax(t0_smpl,dim=1)
            
           
            vacc += torch.sum(acc_mask)
            numel += t0_smpl.shape[0]
    
    return vloss / numel, vacc / numel


def main(args):
    # general preparations
    if torch.cuda.is_available():
        args.device = torch.device(args.device) 
    else:
        args.device = torch.device('cpu')
    
    
    
    logpath = 'logs/log-' + args['basename'] + args.call_prefix + '.txt'
    logpath = os.path.join(args.res_path, logpath)
    hvae_path = 'models/' + args.basename + args.call_prefix
    hvae_path = os.path.join(args.res_path, hvae_path)
    viz_path = 'images/' + args.basename + args.call_prefix 
    viz_path = os.path.join(args.res_path, viz_path)
    stats_path = 'logs/stats-' + args.basename + args.call_prefix + '-enc.npz'
    stats_path = os.path.join(args.res_path, stats_path)
    summary_path = os.path.join(args.res_path,args.basename + args.call_prefix +'-tensorboard_summary')

    # Create outdir and log 
    if not os.path.isdir(args.res_path):
        os.mkdir(args.res_path)
        os.mkdir(os.path.join(args.res_path,'logs'))
        os.mkdir(os.path.join(args.res_path,'models'))
        os.mkdir(os.path.join(args.res_path,'images'))
        os.mkdir(summary_path)

    writer = SummaryWriter(summary_path)    
    
    # prepare data loaders
    t = transforms.ToTensor()
    
    train_dataset = datasets.CIFAR10(args.data_path, train=True, download=False, 
                                transform=t)
    train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=args.batch_workers)
    
    num_workers = os.cpu_count() 
    if 'sched_getaffinity' in dir(os):
        num_workers = len(os.sched_getaffinity(0)) - 2
    
    val_dataset = datasets.CIFAR10(args.data_path, train=False, download=False,
                              transform=t)
    val_dataloader = DataLoader(val_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=args.batch_workers)

    print('# Starting', file=open(logpath, 'w'), flush=True)

    # parse weights to cfg dict 
    

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
    save_period = 1
    niterations = args.niterations
    count = 0
    start_acc = True
    acc_dt = 0.0
    acc_et = 0.0
    best_val_acc = 0


    num_of_batches_seen = 0
    while True:
        # train epoch
        x0_smpl = None
        bar = Bar('Training', max=len(train_dataloader))
        for (xl, tl) in train_dataloader:
            # Iterate over the end if necessary (Can be used with different sizes of dataloaders)
            
            x0_smpl = xl.to(args.device)
            t0_smpl = nn.functional.one_hot(tl, num_classes=10).to(args.device, dtype=torch.float).to(args.device, dtype=torch.float) 
            t0_smpl = t0_smpl.reshape(*t0_smpl.shape,1,1)
            # ====== decoder learn step (unsupervised) ======
              
            
            #nabla_{theta} on MC estimator of  E_pi(x,z0) E_q(z_{>0}|x,z0) log(p(x,z))
            # if argument z0 is a  list of len n, the first n z0 are overidden by the input and rest is 
            # sampled from z0
            
            dt = hvae.decoder_learn_step(x0_smpl,z0=[t0_smpl])
            
            # =====  encoder learn step (unsupervised) ======

            #nabla_{phi} on MC estimator of  E_pi(z) log_p log(q(z|x)) 
            zu0_smpl = hvae.z0_prior_sample(hvae.z0_shape) 
            zu0_smpl[0] = t0_smpl
            et = hvae.encoder_learn_step(zu0_smpl)

            # accumulate data terms
            if start_acc:
                acc_dt = dt.item()
                acc_et = et.item()
                start_acc = False

            acc_dt = acc_dt * 0.999 + dt.item() * 0.001
            acc_et = acc_et * 0.999 + et.item() * 0.001
            

            writer.add_scalars('Loss terms',{'dt' : dt.item(),
                                         'ewa_dt' : acc_dt,
                                         'et' : et.item(),
                                         'ewa_et' : acc_et,
                                         },num_of_batches_seen)

            
            num_of_batches_seen += 1
            bar.suffix  = f"#({num_of_batches_seen%len(train_dataloader)}/{len(train_dataloader)})#({count}/{niterations})|{bar.elapsed_td}|ETA:{bar.eta_td}|"+\
                    f"ewa_dt:{acc_dt:.4f}|ewa_et :{acc_et:.4f}|"
            bar.next()
            break
        bar.finish()

        if (count % log_period == log_period-1) or (count == niterations-1):
            jsdivs = hvae.sblock_list[0].jsdivs
            cat_jsdivs =  jsdivs[0].mean().item()
            bin_jsdivs =  jsdivs[1].mean().item()
            strtoprint = 'epoch: '+ str(count + epoch)
            strtoprint += ' dt: {:.3}'.format(-acc_dt)
            strtoprint += ' et: {:.3}'.format(-acc_et)

            trn_loss, trn_acc = validate(hvae,train_dataloader,args.device)

            strtoprint += ' trnloss: {:.4}'.format(trn_loss)
            strtoprint += ' trnacc: {:.4}'.format(trn_acc)
            
            val_loss, val_acc = validate(hvae, val_dataloader, args.device)
            strtoprint += ' valloss: {:.4}'.format(val_loss)
            strtoprint += ' valacc: {:.4}'.format(val_acc)

            writer.add_scalars('Accuracy',{'trn_acc' : trn_acc,
                                         'val_acc' : val_acc},count)

            writer.add_scalars('Loss',{'trn_loss' : trn_loss,
                                         'val_loss' : val_loss},count)
                                         

            
            print(strtoprint, file=open(logpath, 'a'), flush=True)


        if (save_period and ((count % save_period == save_period - 1) or (count == (args.epochs-1))) ) or val_acc > best_val_acc:
            print('# Saving models ...', file=open(logpath, 'a'), flush=True)
            checkpoint = {
                'model_state_dict': hvae.state_dict(),
                'enc_optimizer_state_dict': hvae.enc_optimizer.state_dict(),
                'dec_optimizer_state_dict': hvae.dec_optimizer.state_dict(),
                'epoch': (count + epoch)}

            h_path = hvae_path + f'-e{count}-a{100*trn_acc:2.0f}-m' + '.pt'
            torch.save(checkpoint, h_path)

            # save encoder stats
            hvae.save_stats(stats_path)

            # update best_val_acc
            best_val_acc = val_acc
               
            # save a few reconstructions
            with torch.no_grad():
                # two batches (sup & unsup)
                n = min(8,xl.shape[0])

                x0 = x0_smpl[0:n]
                t0 = t0_smpl[0:n]
                
                # ====== supervised visualization =================================

                # reconstruction
                _, probs = hvae.posterior_sample(x0,expanded=True)
                x1 = probs[-1] # img
                x4 = probs[0] # z0 (classes)




                
                
                z0_smpl = hvae.z0_prior_sample(hvae.z0_shape) # random sample from latent 
                print(f"{z0_smpl[0].shape=},{z0_smpl[1].shape=}")
                z0_smpl[0] = t0
                print(f"{z0_smpl[0].shape=},{z0_smpl[1].shape=}")

                (x3, x2) = hvae.prior_sample(z0_smpl)
                
                # continue sampling (limiting marginal distrib)
                for _ in range(100):
                    (z_all, _) = hvae.posterior_sample(x3, expanded=True)
                    z_all[0][0] = t0
                    (x3, probs) = hvae.posterior_sample(x3, z0=z_all[0])
                
                x3 = probs[-1]

 
                # x0 & u0 original images

                # z0 latent code 
                # x1  reconstructions from image 
                # x2  reconstructions from latent 
                # x3  limiting reconstruction of x2 
                
                vis = torch.cat((x0, x1, x2, x3))        
                vis.clamp_(0.0, 1.0)
                
                images = vutils.make_grid(vis,nrow=n,normalize=True,scale_each=True)

                writer.add_image(f'img-e{count}-a{val_acc*100:2.0f}', images, count)
                vutils.save_image(images,viz_path +f'-img-e{count}-a{val_acc*100:2.0f}.png')

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
        cfg_name = 'config-2.yaml'
    with open(cfg_name, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    ed = EasyDict(cfg)
    main(ed)
