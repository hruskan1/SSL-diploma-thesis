"""train.py Implementation of the training process. Executable from the command line"""
import argparse
import os
from typing import Callable,Union
from easydict import EasyDict
import yaml
import kornia as K
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms
import mixmatch
import ramps
import utils 
import losses
import datasets
import models
import wide_resnet
import transformations as custom_transforms
from datetime import datetime
import time 
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from progress.bar import Bar as Bar

import ref





def train_one_epoch(model:nn.Module,
                    ewa_model: nn.Module,
                    opt:torch.optim.Optimizer,
                    lr_scheduler:Optional[torch.optim.lr_scheduler._LRScheduler],
                    labeled_dataloader:tv.datasets, 
                    unlabeled_dataloader:tv.datasets,
                    transform:nn.Module,
                    args: EasyDict,
                    metrics: EasyDict,
                    writer: SummaryWriter):
    """
    Training function
    """
    try: 
        ewa_loss = metrics['train_criterion_ewa'][-1]
    except IndexError:
        ewa_loss = 0
    
    model.train()
    labeled_train_iter = iter(labeled_dataloader)
    unlabeled_train_iter = iter(unlabeled_dataloader)
    k = 0 

    bar = Bar('Training', max=args.kimg//args.batch_size + 1)

    while k < args.kimg:
    
        # Iterate over the end if necessary (Can be used with different sizes of dataloaders)
        try:
            data_l, labels = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_dataloader)
            data_l, labels = next(labeled_train_iter)

        try:
            data_u,_ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_dataloader)
            data_u,_ = next(unlabeled_train_iter)

        data_l = data_l.to(args.device)
        labels = labels.to(args.device)
        data_u = data_u.to(args.device)

        # Corner case (batches with different sizes, namely for iregular last batch)
        current_batch_size = min(data_l.shape[0],data_u.shape[0])
        
        data_l = data_l[:current_batch_size]
        labels = labels[:current_batch_size]
        data_u = data_u[:current_batch_size]

        with torch.no_grad():
            ## Maybe not disable?
            #model.eval() # Not used in https://github.com/YU1ut/MixMatch-pytorch
            l_batch,u_batch = mixmatch.mixmatch(labeled_batch=data_l,
                                                    labels=labels,
                                                    unlabeled_batch=data_u,
                                                    clf=model,
                                                    augumentation=transform,
                                                    K=args.K,
                                                    T=args.temperature,
                                                    alpha=args.alpha
                                                    )
            
        x = torch.cat([l_batch[0],u_batch[0]],dim=0)
        targets_l,targets_u = l_batch[1],u_batch[1] 


        # Interleave labeled and unlabeled samples between batches to obtain correct batchnorm calculation
        x_splitted = list(torch.split(x, current_batch_size))
        x_splitted = mixmatch.interleave(x_splitted, current_batch_size)
        
        # Forward 
        model.train() 
        logits = [model(x_splitted[0])]
        for x in x_splitted[1:]:
            logits.append(model(x))

        # Put interleaved samples back
        logits = mixmatch.interleave(logits, current_batch_size)
        logits_l = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        # Losses
        loss_supervised = losses.soft_cross_entropy(logits_l,targets_l,reduction='mean')
        loss_unsupervised = losses.mse_softmax(logits_u,targets_u,reduction='mean')
        
        lam_u = ramps.linear_rampup(current = (args.current_count + k/args.kimg), rampup_length = args.rampup_length) * args.lambda_u
        loss = loss_supervised + lam_u * loss_unsupervised

        # SGD
        opt.zero_grad()
        loss.backward()
        opt.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if args.mean_teacher_coef: 
            ewa_model.update_weights(model)

        # Ewa loss
        if (args.current_count == 0 and k == 0 and ewa_loss == 0):
            ewa_loss = loss        
        else:
            ewa_loss = args.loss_ewa_coef * ewa_loss + (1 - args.loss_ewa_coef) * loss
        
        # Save loss (every time):
        metrics['lambda_u'] = np.append(metrics['lambda_u'],lam_u)
        metrics['train_criterion'] = np.append(metrics['train_criterion'],loss.detach().cpu().numpy())
        metrics['train_criterrion_ewa'] = np.append(metrics['train_criterion'],ewa_loss.detach().cpu().numpy())

        writer.add_scalars('Training loss',{'L':loss,'ewa_L':ewa_loss,'Lx': loss_supervised,'Lu' : loss_unsupervised,'lam_u': lam_u},global_step=len(metrics['lambda_u'])-1)

        k += current_batch_size

       
        bar.suffix  = f"#({k}/{args.kimg})#({args.current_count}/{args.epochs})|{bar.elapsed_td}|ETA:{bar.eta_td}|"+\
                      f"L_ewa:{ewa_loss:.4f}|L:{loss:.4f}|Lx:{loss_supervised:.4f}|Lu:{loss_unsupervised:.4f}|lam: {lam_u:.4f}" +\
                      f""             
        bar.next()
    bar.finish()
    
    return metrics


def train(model:nn.Module,
          opt:torch.optim.Optimizer,
          lr_scheduler:Optional[torch.optim.lr_scheduler._LRScheduler],
          eval_loss_fn:Callable,
          labeled_dataloader:tv.datasets, 
          unlabeled_dataloader:tv.datasets,
          validation_dataloader:tv.datasets,
          test_dataloader:tv.datasets,
          transform:nn.Module,
          args: EasyDict,
          metrics: EasyDict,
          writer: SummaryWriter):
    """
    Training function
    """
    def inner_compute_metrics(model,metrics,writer,step):
            """Inner function to compute metrics"""
            ls_val, acc_val = wide_resnet.evaluate(model, eval_loss_fn, validation_dataloader, args.device)
            metrics['val_loss'] = np.append(metrics['val_loss'],ls_val.detach().cpu().numpy())
            metrics['val_acc'] = np.append(metrics['val_acc'], acc_val.detach().cpu().numpy())
                
            ls_trn, acc_trn = wide_resnet.evaluate(model, eval_loss_fn, labeled_dataloader, args.device)
            metrics['train_loss'] = np.append(metrics['train_loss'], ls_trn.detach().cpu().numpy())
            metrics['train_acc'] = np.append(metrics['train_acc'], acc_trn.detach().cpu().numpy())

            ls_tst,acc_tst = wide_resnet.evaluate(model,eval_loss_fn,test_dataloader,args.device)
            metrics['test_loss'] = np.append(metrics['test_loss'],ls_tst.detach().cpu().numpy())
            metrics['test_acc'] = np.append(metrics['test_acc'],acc_tst.detach().cpu().numpy())

            writer.add_scalars('Accuracy',{'val': acc_val,
                                            'trn': acc_trn,
                                            'tst': acc_tst},step)

            writer.add_scalars('Evaluation loss',{'val': ls_val,
                                                'trn': ls_trn,
                                                'tst': ls_tst},step)
            
            return metrics
    
    
    model = model.to(args.device)
    # Use Teacher if desired
    if args.mean_teacher_coef:
        ewa_model = models.Mean_Teacher(model,args.mean_teacher_coef,weight_decay=args.weight_decay)
        ewa_model = ewa_model.to(args.device)



    
    best_val_acc = 0 
    m = ewa_model if args.mean_teacher_coef else model

    while args.current_count < args.epochs:

        
        if args.current_count == 0: 
            inner_compute_metrics(m,metrics,writer,args.current_count)

        metrics = train_one_epoch(model,m,opt,lr_scheduler,labeled_dataloader,unlabeled_dataloader,transform,args,metrics,writer)
                        
        # Compute validation metrics if validation period 
        args.current_count += 1 
        inner_compute_metrics(m,metrics,writer,args.current_count)


        # Print log if log period
        if ( args.current_count % args.log_period == args.log_period - 1) or (args.current_count == (args.epochs-1) ):
            
            strtoprint =f"epoch: {str(args.current_count)} " + \
                        f"train loss: {metrics['train_loss'][-1]:.4f} " + \
                        f"train acc: {metrics['train_acc'][-1]*100:2.4f} " + \
                        f"val loss: {metrics['val_loss'][-1]:.4f} " + \
                        f"val acc: {metrics['val_acc'][-1]*100:2.4f} " + \
                        f"test loss: {metrics['test_loss'][-1]:.4f} " + \
                        f"test acc: {metrics['test_acc'][-1]*100:2.4f} "  + \
                        f"lam_u {metrics['lambda_u'][-1]:.4f} "  
            
            if args.debug:
                d_ls,d_acc = wide_resnet.evaluate(m, eval_loss_fn, unlabeled_dataloader, args.device)
                strtoprint += f"unlabeled loss: {d_ls:.4f} " + \
                                f"unlabeled acc: {d_acc*100:2.4f}"
                print("M METRICS:")
                _, train_acc = ref.validate(labeled_dataloader, m, eval_loss_fn, args.device, mode='Train Stats')
                val_loss, val_acc = ref.validate(validation_dataloader, m, eval_loss_fn,args.device,  mode='Valid Stats')
                test_loss, test_acc = ref.validate(test_dataloader, m, eval_loss_fn, args.device, mode='Test Stats ')
                                
            print(strtoprint, file=open(args.logpath, 'a'), flush=True)
            print(strtoprint)

        # Save checkpoint if save_period or best so far
        current_val_acc = metrics['val_acc'][-1]
        if args.save_period and ((args.current_count % args.save_period == args.save_period - 1) or (args.current_count == (args.epochs-1))) or current_val_acc >= best_val_acc:

            m_path = args.modelpath + f"-e{str(args.current_count)}" + f"-a{current_val_acc:.2f}" + '-m.pt'
            print(f'# Saving Model : {m_path}', file=open(args.logpath, 'a'), flush=True)
            model_to_save = m.model if isinstance(m,models.Mean_Teacher) else m
            utils.save_checkpoint(args.current_count,metrics,model_to_save,opt,args,m_path)

            if best_val_acc < current_val_acc:
                best_val_acc = current_val_acc
               
    return metrics


if __name__ == '__main__':

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Mixmatch arguments')
    parser.add_argument('-c','--current_count',default=0,type=int,help="Current counter of images. Total length of training is always given by 'e - c' ")
    parser.add_argument('-e','--epochs', default = 1024, type=int, help='Number of epochs to train')
    parser.add_argument('--kimg','--images_per_epoch',default=64,type=int,help='Number of images to see in each epoch (in thousands)')
    parser.add_argument('-K','--K', default = 2, type=int, help='Number of agumentations to apply')
    parser.add_argument('-T','--temperature', default = 0.5, type=float, help='Temperature T for sharpening the distribution, T > 0')
    parser.add_argument('-a','--alpha',default=0.75, type = int, help='Hyperparameter for Beta distribution B(alpha,alpha)')
    parser.add_argument('-lam','--lambda_u',default=75, type = float, help='Weight for loss corresponding to unlabeled data')
    parser.add_argument('-rampup','--rampup_length',default=50, type = int, help='Length of linear ramp which is applied to lambda_u (0->1) in epochs')
    parser.add_argument('-nx','--n_labeled',default=250,type=int,help='Number of labeled samples (Rest is unlabeled)')
    parser.add_argument('-nv','--n_val',default = 500,type=int, help='Number of samples used in validation dataset (the dataset is split into labeled,unlabeled,validation and test if test exists)')
    parser.add_argument('-BS','--batch_size',default=64,type = int,help='mini batch size')
    parser.add_argument('-lr','--learning_rate',default=2e-3,type=float,help='learning rate of Adam Optimizer')
    parser.add_argument('--lr_scheduler',default=False,type=bool,help='Use One Cycle LR scheduler')
    parser.add_argument('--loss_ewa_coef', default = 0.98, type=float, help='weight for exponential weighted average of training loss')
    parser.add_argument('--device',default=1, type = int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset_path',default='./CIFAR10', type=str, help='Root directory for dataset') 
    parser.add_argument('-ewa', '--mean_teacher_coef', default = 0.999, type=float, help='weight for exponential average of mean teacher model. Default is None and Mean teach is not used')
    parser.add_argument('--out', '--output_directory', default=f'./run_{datetime.now().strftime("%d-%m-%Y_%H:%M")}',type=str, help='root directory for results')
    parser.add_argument('--resume', default=None, type=str, help='Path to model which is to be used to resume training')
    parser.add_argument('--from_yml', default=None, type=str, help ='Path to file where the arguments are specified (as a dictionary)')
    parser.add_argument('--log_period', default=1, type=int, help = 'Epoch period for logging the prcoess')
    parser.add_argument('--save_period',default=0,type=int, help = 'Epoch period for saving the model,the model with best val acc is implicitly saved')
    parser.add_argument('--debug',default=False,type=bool, help = 'Enable reporting accuracy and loss on unlabeled data (if data actually have labels')
    parser.add_argument('--seed',default=0,type=int, help ='Manual seed')
    parser.add_argument('--weight_decay',default=0.0004,type=float, help ="Weight decay (applied at each step), Applied only if  'mean_teacher_coef' is not None")

    
    ### Parse the command-line arguments ###
    parsed_args = parser.parse_args()


    # Parse from file
    if parsed_args.from_yml:
        with open(f'{parsed_args.from_yml}', 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            args = EasyDict(cfg)
    else:
        args = EasyDict()


    # Import and override with parameters provided on command line
    for k,v in parsed_args._get_kwargs():
        print(f"{k}:{v}")
        args[k] = v
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get device
    args.device = utils.get_device(args.device)
    
    # Multiply kimg
    args.kimg = args.kimg * 1000

    # Create outdir and log 
    if not os.path.isdir(args.out):
        os.mkdir(args.out)

    args.logpath = os.path.join(args.out,'log.txt')
    args.modelpath = os.path.join(args.out, 'model')

    # Report initlization
    print(f"# Starting at {datetime.now()}",file=open(args.logpath,'w'),flush=True)
    print(f"with args:\n" + "\n".join([f"{key} : {value}" for key,value in args.items()]),file=open(args.logpath,'a'),flush=True)
    if args.resume is not None:
        print(f"resume_with model: {args.resume}",file=open(args.logpath,'a'),flush=True)
    else:
        print(f"No resume point provided, will start from scratch!",file=open(args.logpath,'a'),flush=True)


    print(f"# Starting at {datetime.now()}")

    writer = SummaryWriter(args.out)


    ### Datasets and dataloaders ###
    num_classes = 10 
    labeled_dataloader, unlabeled_dataloader, validation_dataloader,test_dataloader = \
                    datasets.get_CIFAR10(root=args.dataset_path,
                                         n_labeled = args.n_labeled,
                                         n_val = args.n_val,
                                         batch_size = args.batch_size,
                                         download=True,
                                         verbose=True) # remove it later
    

    ### Transformation ###
    k1 = K.augmentation.RandomHorizontalFlip()
    k2 = transforms.Pad(padding=4, padding_mode='reflect')
    k3 = K.augmentation.RandomCrop(size=(32,32))
    # k1 = transforms.RandomHorizontalFlip()
    # k2 = transforms.Pad(padding=4, padding_mode='reflect')
    # k3 = transforms.RandomCrop(size=32)

    img_trans = nn.ModuleList([k1,k2,k3])
    mask_trans = nn.ModuleList() # only for segmentation 
    invert_trans  = nn.ModuleList() # only for segmentation 
    transform = custom_transforms.MyAugmentation(img_trans,mask_trans,invert_trans)


    ### Model,optimizer, LR scheduler, Eval function ###
    model = wide_resnet.WideResNet(num_classes)
    opt = torch.optim.Adam(params=model.parameters(),lr = args.learning_rate)

    if args.lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                        max_lr=3*args.learning_rate,
                                                        epochs=(args.epochs-args.current_count),
                                                        steps_per_epoch=(args.kimg // args.batch_size + 1))
    else:
        lr_scheduler = None

    # eval_loss_fn = losses.kl_divergence
    eval_loss_fn = nn.CrossEntropyLoss()

    # Load previous checkpoint
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"Loading checkpoint : {args.resume}",file=open(args.logpath, 'a'), flush=True)
        print(f"Loading checkpoint : {args.resume}")
        count,metrics,net,opt,net_args = utils.load_checkpoint(args.device,model,opt,args.resume) 
    else:
        print("Creating new network!",file=open(args.logpath, 'a'), flush=True)
        print(f"Creating new network")
        
        metrics = EasyDict()
        metrics['train_criterion'] = np.empty(0)
        metrics['train_criterion_ewa'] = np.empty(0)
        metrics['val_loss'] = np.empty(0)
        metrics['val_acc'] = np.empty(0)
        metrics['train_loss'] = np.empty(0)
        metrics['train_acc'] = np.empty(0)
        metrics['test_loss'] = np.empty(0)
        metrics['test_acc'] = np.empty(0)
        metrics['lambda_u'] = np.empty(0)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0),file=open(args.logpath, 'a'), flush=True)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    metrics = train(model=model,
                    opt = opt,
                    lr_scheduler = lr_scheduler,
                    eval_loss_fn = eval_loss_fn,
                    labeled_dataloader=labeled_dataloader,
                    unlabeled_dataloader=unlabeled_dataloader,
                    validation_dataloader=validation_dataloader,
                    test_dataloader=test_dataloader,
                    transform=transform,
                    args=args,
                    metrics=metrics,
                    writer=writer)
    
    







        

