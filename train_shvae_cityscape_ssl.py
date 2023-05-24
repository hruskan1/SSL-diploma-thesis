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
from src.mixmatch.datasets import get_CityScape,CityScapeDataset
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter


def validate(mod: HVAE, loader: DataLoader, device:torch.cuda.device,ignore_class:int=0):
    """
    Function to evalute model on data in loader
    
    ignore_class:int  class which should be ignored during evaluation
    """
    num_classes = mod.encoder_activations(next(iter(loader))[0].to(device))[0][0].shape[1]

    print(f"DEBUG,REMOVE ME: {num_classes=}")
    weights = torch.ones(num_classes)
    weights = weights[ignore_class] = 0
    cel = nn.CrossEntropyLoss(reduction='none',weight = weights)
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
            vloss += cel(scores, t0_smpl).sum().item()

            acc_mask = torch.argmax(scores,dim=1) == torch.argmax(t0_smpl,dim=1)
            
            # ignore targets with class "ignore_class" 
            igonre_mask = torch.argmax(t0_smpl,dim=1) == ignore_class
            acc_mask[igonre_mask]  = 0
            
            vacc += torch.sum(acc_mask)
            numel += acc_mask.numel() - igonre_mask.sum()
    
    return vloss / numel, vacc / numel


def evaluate_IoU(model:HVAE, dataloader:DataLoader,weights:torch.Tensor)->Tuple[float,torch.Tensor]:
    """
    Computes the average IoU for each class over the entire dataset.

    :param model: A PyTorch model.
    :param dataloader: A PyTorch dataloader.
    :weights: weights for the class iou
    :return: Tuple(average_iou,class_iou), float and tensor containg IoU on dataset for each class [C]
    """
    model.eval()
    device = next(model.parameters()).device

    # get output from model and through that obtain number of classes
    num_classes = model.encoder_activations(next(iter(dataloader))[0].to(device))[0][0].shape[1]

    # Initialize counters for each class
    intersection_total = torch.zeros(num_classes, dtype=torch.float32, device=device)
    union_total = torch.zeros(num_classes, dtype=torch.float32, device=device)

    with torch.no_grad():
        for idx,(inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Compute predictions
            outputs = model.encoder_activations(inputs)[0][0]
            preds = torch.argmax(outputs, dim=1)
            onehot_preds = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()


            # Update the total intersection and union for each class
            current_intersection = torch.sum(onehot_preds * targets, dim=(0, 2, 3))
            intersection_total += current_intersection
            union_total += torch.sum(onehot_preds + targets, dim=(0, 2, 3)) - current_intersection
            
            _avg_iou = torch.mean( (intersection_total + 1e-15) / (union_total + 1e-15) )
            

    # Compute the average IoU score for each class
    class_iou = (intersection_total + 1e-15) / (union_total + 1e-15)
    weighted_iou = class_iou * weights
    n = torch.sum( (weighted_iou > 0) )
    average_iou = torch.sum(weighted_iou) / n

    # Return the results as a dictionary
    return average_iou,class_iou

def main(args):
    # general preparations
    if torch.cuda.is_available():
        args.device = torch.device(args.device) # hot fix 
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
    os.makedirs(args.res_path,exist_ok=True)
    os.makedirs(os.path.join(args.res_path,'logs'),exist_ok=True)
    os.makedirs(os.path.join(args.res_path,'models'),exist_ok=True)
    os.makedirs(os.path.join(args.res_path,'images'),exist_ok=True)
    os.makedirs(summary_path,exist_ok=False)

    writer = SummaryWriter(summary_path)    
    
    # prepare data loaders
    
    labeled_dataloader, unlabeled_dataloader, validation_dataloader,test_dataloader = get_CityScape(root=args.dataset_path,
                                                                                                    n_labeled = args.n_labeled,
                                                                                                    n_val = args.n_val,
                                                                                                    batch_size = args.batch_size,
                                                                                                    mode = 'fine', # change if you want to
                                                                                                    size = args.img_size,
                                                                                                    target_type='semantic',
                                                                                                    verbose=True
                                                                                                    )

    print('# Starting', file=open(logpath, 'w'), flush=True)

    # parse weights to cfg dict 

    args.blocks[0].blocks[0].class_weights = torch.Tensor(CityScapeDataset.class_weights).to(args.device)
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
    best_val_acc = 0

    labeled_train_iter = iter(labeled_dataloader)
    unlabeled_train_iter = iter(unlabeled_dataloader)

    #print(f"{hvae.sblock_list=}")

    num_of_batches_seen = 0
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
                xu,tu = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_dataloader)
                xu,tu = next(unlabeled_train_iter)

            current_batch_size = min(xl.shape[0],xu.shape[0])
        
            xl = xl[:current_batch_size]
            xu = xu[:current_batch_size]
            tl = tl[:current_batch_size]

            k_imgs_seen += current_batch_size
            
        
           
            xl  = xl.to(args.device)
            tl  = tl.to(args.device, dtype=torch.float) 
            xu  = xu.to(args.device)
            tu  = tu.to(args.device, dtype=torch.float)

            # =====  decoder learn step (supervised learning) ====
            
            #nabla_{theta} on MC estimator of  E_pi(x,z0) E_q(z_{>0}|x,z0) log(p(x,z))

            sup_dt = hvae.decoder_learn_step(xl, z0=[tl])
            
            # =====  encoder learn step (supervised learning) ====

            # nabla_{theta} of monte carlo estimatior of E_pi(x,z0) log(q_tilde_{phi}(z_0|x)) 
            z0l_smpl = hvae.z0_prior_sample(hvae.z0_shape)
            z0l_smpl[0] = tl 
            sup_et = hvae.encoder_supervised_learn_step(xl,z0=z0l_smpl)
            
            # unsupervised batch 
            # ====== decoder learn step (unsupervised) ======

            # nabla_{theta} on MC estimator of  E_pi(x) E_q(z|x) log(p(x,z))
            unsup_dt = hvae.decoder_learn_step(xu)

            # =====  encoder learn step (unsupervised) ======

            # E_pi(z0) E_p(x,z_{>0}|z0) log(q_{theta,phi}(z|x))
            zu0_smpl = hvae.z0_prior_sample(hvae.z0_shape) # random sample from latent 
            zu0_smpl[0] = tu 
            unsup_et = hvae.encoder_learn_step(zu0_smpl)

            # accumulate data terms
            if start_acc:
                sup_acc_dt = sup_dt.item()
                sup_acc_et = sup_et.item()
                unsup_acc_dt = unsup_dt.item()
                unsup_acc_et = unsup_et.item()
                start_acc = False

            sup_acc_dt = sup_acc_dt * 0.999 + sup_dt.item() * 0.001
            sup_acc_et = sup_acc_et * 0.999 + sup_et.item() * 0.001
            unsup_acc_dt = unsup_acc_dt * 0.999 + unsup_dt.item() * 0.001
            unsup_acc_et = unsup_acc_et * 0.999 + unsup_et.item() * 0.001

            writer.add_scalars('Loss terms',{'sup_dt' : sup_dt.item(),
                                         'ewa_sup_dt' : sup_acc_dt,
                                         'sup_et' : sup_et.item(),
                                         'ewa_sup_et' : sup_acc_et,
                                         'unsup_dt' : unsup_dt.item(),
                                         'ewa_unsup_dt' : unsup_acc_dt,
                                         'unsup_et' : unsup_et.item(),
                                         'ewa_unsup_et' :unsup_acc_et},num_of_batches_seen)

            
            print(f"\r{count=}|({k_imgs_seen}/{args.kimg})",end='')
            num_of_batches_seen += 1

        if (count % log_period == log_period-1) or (count == niterations-1):
            strtoprint = 'epoch: '+ str(count + epoch)
            strtoprint += ' sup_dt: {:.3}'.format(-sup_acc_dt)
            strtoprint += ' sup_et: {:.3}'.format(-sup_acc_et)
            strtoprint += ' unsup_dt: {:.3}'.format(-unsup_acc_dt)
            strtoprint += ' unsup_et: {:.3}'.format(-unsup_acc_et)

            trn_loss, trn_acc = validate(hvae,labeled_dataloader,args.device,ignore_class=0)

            strtoprint += ' trnloss: {:.4}'.format(trn_loss)
            strtoprint += ' trnacc: {:.4}'.format(trn_acc)
            
            val_loss, val_acc = validate(hvae, test_dataloader, args.device,ignore_class=0)
            strtoprint += ' valloss: {:.4}'.format(val_loss)
            strtoprint += ' valacc: {:.4}'.format(val_acc)

            writer.add_scalars('Accuracy',{'trn_acc' : trn_acc,
                                         'val_acc' : val_acc},count)

            writer.add_scalars('Loss',{'trn_loss' : trn_loss,
                                         'val_loss' : val_loss},count)
                                         

            
            print(strtoprint, file=open(logpath, 'a'), flush=True)


        if (save_period and ((count % save_period == save_period - 1) or (count == (niterations-1))) ) or val_acc > best_val_acc:
            print('# Saving models ...', file=open(logpath, 'a'), flush=True)
            checkpoint = {
                'model_state_dict': hvae.state_dict(),
                'enc_optimizer_state_dict': hvae.enc_optimizer.state_dict(),
                'dec_optimizer_state_dict': hvae.dec_optimizer.state_dict(),
                'epoch': (count + epoch)}

            h_path = hvae_path + f'-e{count}-a{100*val_acc:2.0f}-m' + '.pt'
            torch.save(checkpoint, h_path)

            # save encoder stats
            hvae.save_stats(stats_path)

            # update best_val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # evalute IoU
            # w = torch.Tensor(CityScapeDataset.class_weights).to(args.device)
            # trn_avg_iou,trn_class_iou = evaluate_IoU(hvae,labeled_dataloader, w)
            # val_avg_iou,val_class_iou = evaluate_IoU(hvae,test_dataloader, w)
            # tags = [CityScapeDataset.trainid2names[trainid] for trainid in range(trn_class_iou.shape[0])]
            
            # writer.add_scalars('train class IoU',{tags[i]: trn_class_iou[i] for i in range(trn_class_iou.shape[0])},count)
            # writer.add_scalars('val class IoU',{tags[i]: val_class_iou[i] for i in range(val_class_iou.shape[0])},count)
            # writer.add_scalars('average IoU',{  'trn': trn_avg_iou,
            #                                     'val': val_avg_iou},count)
                
            

            # save a few reconstructions
            with torch.no_grad():
                # two batches (sup & unsup)
                n = min(8,xl.shape[0])

                x0l = xl[0:n]
                x0u = xu[0:n]
                t0l = tl[0:n]
                
                _s = torch.empty((n,*list(hvae.z0_shape[0][1:])))
                _l = torch.empty((n,*list(hvae.z0_shape[1][1:])))
                
                
                z0_shape = (_s.shape,_l.shape)
                
                z0l_smpl = hvae.z0_prior_sample(z0_shape) # random sample from latent 
                z0l_smpl[0] = t0l

                z0u_smpl = hvae.z0_prior_sample(z0_shape) # random sample from latent 

                # ====== supervised visualization =================================

                # reconstruction
                _, probs = hvae.posterior_sample(x0l,expanded=True)
                x1 = probs[-1] # img
                x4 = probs[0][0] # target (segmentation)
                
                
                (x3, x2) = hvae.prior_sample(z0l_smpl)
                
                # continue sampling (limiting marginal distrib)
                for _ in range(100):
                    (z_all, _) = hvae.posterior_sample(x3, expanded=True)
                    z_all[0][0] = t0l
                    (x3, probs) = hvae.posterior_sample(x3, z0=z_all[0])
                x3 = probs[-1]

                # ====== unsupervied visualization ===============================

                # reconstruction
                _, probs = hvae.posterior_sample(x0u,expanded=True)
                u1 = probs[-1] # img
                u4 = probs[0][0] # target (segmentation)

                (u3,u2) = hvae.prior_sample(z0u_smpl)

                for _ in range(100):
                    (u3, _) = hvae.posterior_sample(u3)
                (_, probs) = hvae.posterior_sample(u3)

                u3 = probs[-1]
                # x0 & u0 original images

                # z0x original segmentation 
                # x1 & u1 reconstructions from image 
                # x2 & u2 reconstructions from latent (segmentation or random sample)
                # x3 & u3 limiting reconstruction of x2 & u2 
                # x4 & u4 predicted segmentation for x1 & u1
                
                xvis = torch.cat((x0l, x0u, x1, u1, x2, u2, x3, u3))
                xvis = CityScapeDataset.remove_normalization(xvis)
    
                segvis = torch.cat((x4, u4, t0l))    
                segvis = CityScapeDataset.color_segmentation(torch.argmax(segvis,dim=1,keepdim=True))
                vis = torch.cat([xvis,segvis],dim=0)
            
                vis.clamp_(0.0, 1.0)
                
                images = vutils.make_grid(vis,nrow=2*n,normalize=True,scale_each=True)

                writer.add_image(f'img-e{count}-a{val_acc*100:2.0f}', images, count)
                vutils.save_image(images, viz_path +f'-img-e{count}-a{val_acc*100:2.0f}.png')

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
