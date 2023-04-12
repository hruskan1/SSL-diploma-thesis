"""train.py Implementation of learning process for MixMatch algorithm. The high-level functions to run the process."""
from typing import Callable
from easydict import EasyDict
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from . import mixmatch
from . import ramps
from . import utils 
from . import losses
from .mean_teacher import MeanTeacher 
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from progress.bar import Bar 
from ..models.misc import evaluate,evaluate_IoU,visulize_batch
from ..models.unet.unet import Unet
from ..mixmatch.datasets import CityScapeDataset,get_base_dataset




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
    Trains one epoch of Mixmatch Algorithm process

    Parameters:
    - model: A PyTorch model to train
    - ewa_model: A PyTorch EMA (exponential moving average) model to use (optional)
    - opt: A PyTorch optimizer to use
    - lr_scheduler: A PyTorch learning rate scheduler to use (optional)
    - labeled_dataloader: A PyTorch dataloader for labeled data
    - unlabeled_dataloader: A PyTorch dataloader for unlabeled data
    - transform: A PyTorch transform to apply to the data 
    - args: A dictionary containing various hyperparameters for the algorithm
    - metrics: A dictionary to store training metrics
    - writer: A PyTorch SummaryWriter object to write metrics to (optional)

    Returns:
    - metrics: A dictionary containing training metrics
    """
    try: 
        ewa_loss = metrics['train_criterion_ewa'][-1]
    except IndexError:
        ewa_loss = 0
    
    model.train()
    labeled_train_iter = iter(labeled_dataloader)
    unlabeled_train_iter = iter(unlabeled_dataloader)
    k = 0 

    bar = Bar('Training', max=args.kimg//args.batch_size)

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
          ewa_model:nn.Module,
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
    Trains a PyTorch model for Mixmatch Algorithm. Saves and logs required metrics and others.
    
    Args:
        model (nn.Module): The PyTorch model to train.
        ewa_model (Optional(nn.Module)) The EWA of trained model (Mean Teacher)
        opt (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        lr_scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): The learning rate scheduler to use, if any.
        eval_loss_fn (Callable): The loss function used to evaluate the model during training.
        labeled_dataloader (tv.datasets): The dataloader used to load the labeled training data.
        unlabeled_dataloader (tv.datasets): The dataloader used to load the unlabeled training data.
        validation_dataloader (tv.datasets): The dataloader used to load the validation data.
        test_dataloader (tv.datasets): The dataloader used to load the test data.
        transform (nn.Module): The data augmentation transforms to apply to the input data.
        args (EasyDict): A dictionary of hyperparameters and training settings.
        metrics (EasyDict): A dictionary of evaluation metrics to use during training.
        writer (SummaryWriter): A tensorboard summary writer to log training and validation metrics.
    """
    def inner_compute_metrics(model,metrics,writer,step):
            """Inner function to compute metrics"""
            if validation_dataloader is not None > 0:
                ls_val, acc_val = evaluate(model, eval_loss_fn, validation_dataloader, args.device)
            else:
                acc_val,ls_val = (torch.Tensor([0]),torch.Tensor([0]))
            
            metrics['val_loss'] = np.append(metrics['val_loss'],ls_val.detach().cpu().numpy())
            metrics['val_acc'] = np.append(metrics['val_acc'], acc_val.detach().cpu().numpy())
                
            ls_trn, acc_trn = evaluate(model, eval_loss_fn, labeled_dataloader, args.device)
            metrics['train_loss'] = np.append(metrics['train_loss'], ls_trn.detach().cpu().numpy())
            metrics['train_acc'] = np.append(metrics['train_acc'], acc_trn.detach().cpu().numpy())

            ls_tst,acc_tst = evaluate(model,eval_loss_fn,test_dataloader,args.device)
            metrics['test_loss'] = np.append(metrics['test_loss'],ls_tst.detach().cpu().numpy())
            metrics['test_acc'] = np.append(metrics['test_acc'],acc_tst.detach().cpu().numpy())

            writer.add_scalars('Accuracy',{'val': acc_val,
                                            'trn': acc_trn,
                                            'tst': acc_tst},step)

            writer.add_scalars('Evaluation loss',{'val': ls_val,
                                                'trn': ls_trn,
                                                'tst': ls_tst},step)
            
            # Compute IoU 
            model_to_eval = model.model if isinstance(m,MeanTeacher) else model
            if isinstance(model_to_eval,Unet):
                trn_avg_iou,trn_class_iou = evaluate_IoU(model,labeled_dataloader)

                if validation_dataloader is not None:
                    val_avg_iou,val_class_iou = evaluate_IoU(model_to_eval,validation_dataloader)
                else:
                    val_avg_iou,val_class_iou = trn_avg_iou * 0, trn_class_iou * 0

                tst_avg_iou,tst_class_iou = evaluate_IoU(model_to_eval,test_dataloader)

                writer.add_scalars('average IoU',{  'trn': trn_avg_iou,
                                                    'tst': tst_avg_iou,
                                                    'val': val_avg_iou},step)
                
                
                if isinstance(get_base_dataset(labeled_dataloader),CityScapeDataset):
                    tags = [CityScapeDataset.trainid2names[trainid] for trainid in range(trn_class_iou.shape[0])]
                else: 
                    tags = [str(i) for i in range(trn_class_iou.shape[0])]
                
                writer.add_scalars('train class IoU',{tags[i]: trn_class_iou[i] for i in range(trn_class_iou.shape[0])},step)
                writer.add_scalars('val class IoU',{tags[i]: val_class_iou[i] for i in range(val_class_iou.shape[0])},step)
                writer.add_scalars('test class IoU',{tags[i]: tst_class_iou[i] for i in range(tst_class_iou.shape[0])},step)
                

            
            return metrics
    

    model = model.to(args.device)
    if ewa_model is not None:
        ewa_model = ewa_model.to(args.device)
    
    m = ewa_model if ewa_model is not None else model
    best_val_acc = 0 

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
                try:
                    last_lr = opt.param_groups[0]['lr'][-1] if lr_scheduler is None else lr_scheduler.get_last_lr()[-1]
                except TypeError:
                    last_lr = opt.param_groups[0]['lr'] if lr_scheduler is None else lr_scheduler.get_last_lr()
            
                d_ls,d_acc = evaluate(m, eval_loss_fn, unlabeled_dataloader, args.device)
                strtoprint += f"unlab loss: {d_ls:.4f} " + \
                              f"unlab acc: {d_acc*100:2.4f}" +\
                              f"LR: {last_lr:.6f}"  
                              
            print(strtoprint, file=open(args.logpath, 'a'), flush=True)
            print(strtoprint)

        # Save checkpoint if save_period or best so far
        current_val_acc = metrics['test_acc'][-1]
        if args.save_period and ((args.current_count % args.save_period == args.save_period - 1) or (args.current_count == (args.epochs-1))) or current_val_acc > best_val_acc:

            m_path = args.modelpath + f"-e{str(args.current_count)}" + f"-a{current_val_acc*100:2.0f}" + '-m.pt'
            print(f'# Saving Model : {m_path}', file=open(args.logpath, 'a'), flush=True)
            model_to_save = m.model if isinstance(m,MeanTeacher) else m
            utils.save_checkpoint(args.current_count,metrics,model_to_save,opt,args,m_path)

            best_val_acc = current_val_acc

            # Make visualization if debug and 
            if args.debug and isinstance(model,Unet):
                viz_folder = os.path.join(args.out,'figs')
                visulize_batch(model,labeled_dataloader,transform,writer,viz_folder,f"trn-e{args.current_count}-a{current_val_acc*100:2.0f}")
                if validation_dataloader is not None:
                    visulize_batch(model,validation_dataloader,transform,writer,viz_folder,f"val-e{args.current_count}-a{current_val_acc*100:2.0f}")
                
                visulize_batch(model,test_dataloader,transform,writer,viz_folder,f"tst-e{args.current_count}-a{current_val_acc*100:2.0f}")
                          
    return metrics