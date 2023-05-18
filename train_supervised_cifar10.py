"""train_supervised_cifar10.py Implementation of the training process for supervised scenario. Executable from the command line"""
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
from torch.utils import data


from datetime import datetime
import time 
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar as Bar


from src.supervised.train import train
from src.models.wide_resnet import wide_resnet
from src.mixmatch import utils


if __name__ == '__main__':

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Supervised CIFAR10 arguments')
    parser.add_argument('-c','--current_count',default=0,type=int,help="Current counter of images. Total length of training is always given by 'e - c' ")
    parser.add_argument('-e','--epochs', default = 500, type=int, help='Number of epochs to train')
    parser.add_argument('--WRN_depth', default = 28, type=int, help='Depth of wide resnet')
    parser.add_argument('--WRN_factor', default = 2, type=int, help='Widen factor for resnet')
    parser.add_argument('--WRN_dropout', default = 0.3, type=int, help='dropout rate for resnet')
    parser.add_argument('-BS','--batch_size',default=64,type = int,help='mini batch size')
    parser.add_argument('-lr','--learning_rate',default=None,type=utils._float,help='learning rate of Adam Optimizer')
    parser.add_argument('--lr_scheduler',default=True,type=bool,help='Use One Cycle LR scheduler')
    parser.add_argument('--loss_ewa_coef', default = 0.98, type=utils._restricted_float, help='weight for exponential weighted average of training loss')
    parser.add_argument('--device',default=0, type = int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset_path',default='~/CIFAR10', type=str, help='Root directory for dataset') 
    parser.add_argument('--out', '--output_directory', default=f'../sl_cifar10_run_{datetime.now().strftime("%d-%m-%Y_%H:%M")}',type=str, help='root directory for results')
    parser.add_argument('--resume', default=None, type=utils._str, help='Path to model which is to be used to resume training')
    parser.add_argument('--from_yml', default=None, type=utils._str, help ='Path to file where the arguments are specified (as a dictionary)')
    parser.add_argument('--log_period', default=1, type=int, help = 'Epoch period for logging the prcoess')
    parser.add_argument('--save_period',default=0,type=int, help = 'Epoch period for saving the model,the model with best val acc is implicitly saved')
    parser.add_argument('--debug',default=False,type=bool, help = 'Enable reporting accuracy and loss on unlabeled data (if data actually have labels')
    parser.add_argument('--seed',default=0,type=int, help ='Manual seed')
    parser.add_argument('-nt','--n_total',default=15000,type=int,help="number of images available (maximum is what the dataset contains)")
    

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
    # Mean and standard deviation for CIFAR10
    mean = torch.Tensor([0.4914, 0.4822, 0.4465]) 
    std = torch.Tensor([0.2471, 0.2435, 0.2616])

    _t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=mean,std=std),
                             
                             ]) 
    augumentation = transforms.Compose([K.augmentation.RandomHorizontalFlip(p=0.5),
                                        transforms.Pad(padding=4, padding_mode='reflect'),
                                        K.augmentation.RandomCrop(size=(32,32))])

    train_dataset = tv.datasets.CIFAR10(args.dataset_path, train = True, transform = _t, download = True)
    test_dataset = tv.datasets.CIFAR10(args.dataset_path, train = False, transform = _t, download = True)
    
    if args.n_total < len(train_dataset):
        indicies = torch.randperm(len(train_dataset))[:args.n_total]
        train_dataset = data.Subset(train_dataset,indicies)

    # Taken from here https://stackoverflow.com/a/58748125/1983544
    import os
    num_workers = os.cpu_count() 
    if 'sched_getaffinity' in dir(os):
        num_workers = len(os.sched_getaffinity(0)) - 2

    train_dataloader = data.DataLoader(train_dataset,args.batch_size,shuffle=True,num_workers=num_workers,drop_last=True)
    test_dataloader =  data.DataLoader(test_dataset,args.batch_size,shuffle=False,num_workers=num_workers,drop_last=False)


    ### Model,optimizer, LR scheduler, Eval function ###
    model = wide_resnet.WideResNet(num_classes, depth=args.WRN_depth, widen_factor=args.WRN_factor, dropRate=args.WRN_dropout)
    loss_fn = nn.CrossEntropyLoss()

    # find optimal lr
    if args.learning_rate is None:
        losses, lrs = utils.lr_find(model,
                train_dl = train_dataloader,
                loss_fn = loss_fn,
                args = args,
                transform = None,
                min_lr = 1e-7, max_lr = 100, steps = 50,
                optim_fn=torch.optim.Adam)
    
        args.learning_rate = 1/3 * lrs[torch.argmin(losses,dim=0)]
        print(f"Computed lr: {args.learning_rate}")

    opt = torch.optim.Adam(params=model.parameters(),lr = args.learning_rate)

    if args.lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                        max_lr=3*args.learning_rate,
                                                        epochs=(args.epochs-args.current_count),
                                                        steps_per_epoch=(len(train_dataloader)))
    else:
        lr_scheduler = None

    # Load previous checkpoint
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"Loading checkpoint : {args.resume}",file=open(args.logpath, 'a'), flush=True)
        print(f"Loading checkpoint : {args.resume}")
        # Config args contains old configuration args, i.e. "args"
        count,metrics,net,opt,config_args = utils.load_checkpoint(args.device,model,opt,args.resume) 
        
    else:
        print("Creating new network!",file=open(args.logpath, 'a'), flush=True)
        print(f"Creating new network")
        
        metrics = EasyDict()
        metrics['train_criterion'] = np.empty(0)
        metrics['train_criterion_ewa'] = np.empty(0)
        metrics['train_loss'] = np.empty(0)
        metrics['train_acc'] = np.empty(0)
        metrics['test_loss'] = np.empty(0)
        metrics['test_acc'] = np.empty(0)
        metrics['val_loss'] = np.empty(0)
        metrics['val_acc'] = np.empty(0)


    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0),file=open(args.logpath, 'a'), flush=True)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    metrics = train(model=model,
                    opt = opt,
                    lr_scheduler = lr_scheduler,
                    loss_fn = loss_fn,
                    train_dataloader = train_dataloader,
                    validation_dataloader= None,
                    test_dataloader = test_dataloader,
                    transform  = augumentation,
                    args=args,
                    metrics=metrics,
                    writer=writer)
    
    







        

