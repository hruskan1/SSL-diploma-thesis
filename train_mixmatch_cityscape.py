"""train_mixmatch_cityscape.py Implementation of the training process of MixMatch. Executable from the command line"""
import argparse
import os
from easydict import EasyDict
import yaml
import kornia as K

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils import data
from src.mixmatch import losses
import src.mixmatch.utils as utils
import src.mixmatch.datasets as my_datasets
from src.mixmatch.mean_teacher import MeanTeacher
from src.models.unet import unet
from src.mixmatch.transformations import MyAugmentation
from datetime import datetime

from typing import Optional,Tuple

from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar as Bar
from src.mixmatch.train import train
from src.models.misc import visulize_batch


def parse_cmdline_arguments()-> EasyDict:
    """parse command-line argumetns into EasyDict"""
        # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Mixmatch CityScape arguments')
    parser.add_argument('-c','--current_count',default=0,type=int,help="Current counter of images. Total length of training is always given by 'e - c' ")
    parser.add_argument('-e','--epochs', default = 1024, type=int, help='Number of epochs to train')
    parser.add_argument('--kimg','--images_per_epoch',default=5,type=float,help='Number of images to see in each epoch (in thousands)')
    parser.add_argument('-K','--K', default = 2, type=int, help='Number of agumentations to apply')
    parser.add_argument('-T','--temperature', default = 0.5, type=float, help='Temperature T for sharpening the distribution, T > 0')
    parser.add_argument('-a','--alpha',default = 0.3, type = float, help='Hyperparameter for Beta distribution B(alpha,alpha)')
    parser.add_argument('-lam','--lambda_u',default= 116, type = float, help='Weight for loss corresponding to unlabeled data')
    parser.add_argument('-rampup','--rampup_length',default = 1024, type = int, help='Length of linear ramp which is applied to lambda_u (0->1) in epochs')
    parser.add_argument('-nx','--n_labeled',default = 500,type=int,help='Number of labeled samples (Rest is unlabeled)')
    parser.add_argument('-nv','--n_val',default = 0,type=int, help='Number of samples used in validation dataset (the dataset is split into labeled,unlabeled,validation and test if test exists)')
    parser.add_argument('-BS','--batch_size',default=2,type = int,help='mini batch size')
    parser.add_argument('-lr','--learning_rate',default=2e-3,type=float,help='learning rate of Adam Optimizer')
    parser.add_argument('--lr_scheduler',default=False,type=bool,help='Use One Cycle LR scheduler')
    parser.add_argument('--loss_ewa_coef', default = 0.98, type=utils._restricted_float, help='weight for exponential weighted average of training loss')
    parser.add_argument('--device',default=1, type = int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset_path',default='/datagrid/public_datasets/CityScapes', type=str, help='Root directory for dataset') 
    parser.add_argument('-ewa', '--mean_teacher_coef', default = 0.999, type=utils._restricted_float, help='weight for exponential average of mean teacher model. Default is None and Mean teach is not used')
    parser.add_argument('--out', '--output_directory', default=f'./sl_city_run@mix_{datetime.now().strftime("%d-%m-%Y_%H:%M")}',type=str, help='root directory for results')
    parser.add_argument('--resume', default=None, type=utils._str, help='Path to model which is to be used to resume training')
    parser.add_argument('--from_yml', default=None, type=utils._str, help ='Path to file where the arguments are specified (as a dictionary)')
    parser.add_argument('--log_period', default=1, type=int, help = 'Epoch period for logging the prcoess')
    parser.add_argument('--save_period',default=0,type=int, help = 'Epoch period for saving the model,the model with best val acc is implicitly saved')
    parser.add_argument('--debug',default=True,type=bool, help = 'Enable reporting accuracy and loss on unlabeled data (if data actually have labels')
    parser.add_argument('--seed',default=0,type=int, help ='Manual seed')
    parser.add_argument('--weight_decay',default=4e-5,type=float, help ="Weight decay (applied at each step), Applied only if  'mean_teacher_coef' is not None")
    parser.add_argument('--model_architecture', default='./src/models/unet/large_size.yaml', type=str, help = 'Path to the model architecture (yaml)')
    parser.add_argument('--img_size', default = [128,256], type=int,nargs="+", help = '(H:int,W:int) shape of image' )
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
        args[k] = v

    # Get architecture of u-net
    with open(f"{args.model_architecture}",'r') as ymlfile:
        cfg = yaml.load(ymlfile,Loader=yaml.FullLoader)
        args.model_architecture = EasyDict(cfg)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get device
    args.device = utils.get_device(args.device)
    
    # Multiply kimg
    args.kimg = int(args.kimg * 1000)

    return args


def prepare_transformation(size:Tuple[int,int])-> MyAugmentation:
    ### Transformation ###
    k1 = transforms.Pad(padding=4, padding_mode='reflect')
    k2 = K.augmentation.RandomCrop(size=size,same_on_batch=True) #nessecary for mixmatch 
    k3 = K.augmentation.RandomHorizontalFlip()
    
    img_trans = nn.ModuleList([k1,k2,k3])
    mask_trans = nn.ModuleList([k1,k2,k3]) # only for segmentation 
    invert_trans  = nn.ModuleList([k3]) # only for segmentation 
    augumentation = MyAugmentation(img_trans,mask_trans,invert_trans)

    return augumentation



if __name__ == '__main__':

    args = parse_cmdline_arguments()

    # Create outdir and log 
    args.tensorboardpath =os.path.join(args.out,'tensorboard_summary')
    args.logpath = os.path.join(args.out,'log.txt')
    args.modelpath = os.path.join(args.out, 'model')
 
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
        os.mkdir(args.tensorboardpath)
        
    writer = SummaryWriter(args.tensorboardpath)

    # Report initlization
    print(f"# Starting at {datetime.now()}",file=open(args.logpath,'w'),flush=True)
    print(f"with args:\n" + "\n".join([f"{key} : {value}" for key,value in args.items()]),file=open(args.logpath,'a'),flush=True)
    if args.resume is not None:
        print(f"resume_with model: {args.resume}",file=open(args.logpath,'a'),flush=True)
    else:
        print(f"No resume point provided, will start from scratch!",file=open(args.logpath,'a'),flush=True)


    print(f"# Starting at {datetime.now()}")

    ### Augumentation, Datasets and dataloaders ###
    img_size = args.img_size

    augumentation = prepare_transformation(img_size)
    labeled_dataloader, unlabeled_dataloader, validation_dataloader,test_dataloader =\
                                                my_datasets.get_CityScape(root=args.dataset_path,
                                                                            n_labeled = args.n_labeled,
                                                                            n_val = args.n_val,
                                                                            batch_size = args.batch_size,
                                                                            mode = 'fine', # change if you want to
                                                                            size = img_size,
                                                                            target_type='semantic',
                                                                            verbose=True
                                                                            )
     
    

    ### Model,optimizer, LR scheduler, Eval function ###
    model = unet.Unet(args.model_architecture)
    args.class_weights = torch.Tensor(my_datasets.CityScapeDataset.class_weights).to(args.device)
    
    eval_loss_fn = losses.SoftCrossEntropy(weight=args.class_weights,reduction='none')
    #eval_loss_fn = nn.CrossEntropyLoss()

    # find optimal lr
    if args.learning_rate is None:
        print("Searching for learning rate!")
        losses, lrs = utils.lr_find(model,
                train_dl = labeled_dataloader,
                loss_fn = eval_loss_fn,
                args = args,
                transform = None,
                min_lr = 1e-7, max_lr = 100, steps = 50,
                optim_fn = torch.optim.Adam,
                verbose=True)
    
        args.learning_rate = 1/3 * lrs[torch.argmin(losses,dim=0)]
        print(f"Computed lr: {args.learning_rate}")
    
    opt = torch.optim.Adam(params=model.parameters(),lr = args.learning_rate)

    if args.lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                        max_lr=3*args.learning_rate,
                                                        epochs=(args.epochs-args.current_count),
                                                        steps_per_epoch=(args.kimg // args.batch_size))
    else:
        lr_scheduler = None

    # Use Teacher if desired
    if args.mean_teacher_coef:
        ewa_model = MeanTeacher(model,args.mean_teacher_coef,weight_decay=args.weight_decay)
    else:
        ewa_model = None

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


    metrics = train(model = model,
                    ewa_model = ewa_model,
                    opt = opt,
                    lr_scheduler = lr_scheduler,
                    eval_loss_fn = eval_loss_fn,
                    labeled_dataloader = labeled_dataloader,
                    unlabeled_dataloader = unlabeled_dataloader,
                    validation_dataloader = validation_dataloader,
                    test_dataloader = test_dataloader,
                    transform = augumentation,
                    args = args,
                    metrics = metrics,
                    writer = writer)
    
    







        

