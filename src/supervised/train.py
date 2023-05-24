"""train.py Module contating standard supervised learning """

import os
import torch
import torch.nn as nn
import torchvision as tv
import numpy as np
from easydict import EasyDict
from progress.bar import Bar
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from typing import Callable,Tuple,Optional
from ..models.misc import evaluate,evaluate_IoU,ensure_onehot,visulize_batch
from ..mixmatch import utils
from ..models.unet.unet import Unet
from ..mixmatch.datasets import CityScapeDataset,get_base_dataset

def train(model:nn.Module,
          opt:torch.optim.Optimizer,
          lr_scheduler:Optional[torch.optim.lr_scheduler._LRScheduler],
          loss_fn:Callable,
          train_dataloader:tv.datasets,
          validation_dataloader:tv.datasets,
          test_dataloader:tv.datasets,
          transform:Optional[nn.Module],
          args: EasyDict,
          metrics: EasyDict,
          writer: SummaryWriter):
    """
    Trains a PyTorch model using the labeled training dataloader, and evaluates it
    on the validation and test sets. Supports data augmentation using the `transform` argument.
    
    Args:
        model (nn.Module): The PyTorch model to train.
        ewa_model (Optional(nn.Module)) The EWA of trained model (Mean Teacher)
        opt (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        lr_scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): The learning rate scheduler to use, if any.
        loss_fn (Callable): The loss function used to evaluate the model during training.
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
            """Inner function to compute metrics - can be time demanding, especially in segmentation"""
            if validation_dataloader is not None:
                ls_val, acc_val = evaluate(model, loss_fn, validation_dataloader, args.device)
            else:
                ls_val, acc_val = (torch.Tensor([0]),torch.Tensor([0]))
            
            metrics['val_loss'] = np.append(metrics['val_loss'],ls_val.detach().cpu().numpy())
            metrics['val_acc'] = np.append(metrics['val_acc'], acc_val.detach().cpu().numpy())
   
            ls_trn, acc_trn = evaluate(model, loss_fn, train_dataloader, args.device)
            metrics['train_loss'] = np.append(metrics['train_loss'], ls_trn.detach().cpu().numpy())
            metrics['train_acc'] = np.append(metrics['train_acc'], acc_trn.detach().cpu().numpy())

            ls_tst,acc_tst = evaluate(model,loss_fn,test_dataloader,args.device)
            metrics['test_loss'] = np.append(metrics['test_loss'],ls_tst.detach().cpu().numpy())
            metrics['test_acc'] = np.append(metrics['test_acc'],acc_tst.detach().cpu().numpy())

            writer.add_scalars('Accuracy',{ 'trn': acc_trn,
                                            'tst': acc_tst,
                                            'val': acc_val},step)

            writer.add_scalars('Evaluation loss',{'trn': ls_trn,
                                                  'tst': ls_tst,
                                                  'val': ls_val},step)
            
            # Compute IoU 
            if isinstance(model,Unet):
                w = args.class_weights > 0
                trn_avg_iou,trn_class_iou = evaluate_IoU(model,train_dataloader,w)

                if validation_dataloader is not None:
                    val_avg_iou,val_class_iou = evaluate_IoU(model,validation_dataloader,w)
                else:
                    val_avg_iou,val_class_iou = trn_avg_iou * 0, trn_class_iou * 0

                tst_avg_iou,tst_class_iou = evaluate_IoU(model,test_dataloader,w)

                writer.add_scalars('average IoU',{  'trn': trn_avg_iou,
                                                    'tst': tst_avg_iou,
                                                    'val': val_avg_iou},step)
                if isinstance(get_base_dataset(train_dataloader),CityScapeDataset):
                    tags = [CityScapeDataset.trainid2names[trainid] for trainid in range(trn_class_iou.shape[0])]
                else: 
                    tags = [str(i) for i in range(trn_class_iou.shape[0])]

                writer.add_scalars('train class IoU',{tags[i]: trn_class_iou[i] for i in range(trn_class_iou.shape[0])},step)
                writer.add_scalars('val class IoU',{tags[i]: val_class_iou[i] for i in range(val_class_iou.shape[0])},step)
                writer.add_scalars('test class IoU',{tags[i]: tst_class_iou[i] for i in range(tst_class_iou.shape[0])},step)
                

            
            return metrics
    
    
    model = model.to(args.device)
    best_val_acc = 0 
   
    while args.current_count < args.epochs:

        
        if args.current_count == 0: 
            inner_compute_metrics(model,metrics,writer,args.current_count)


        try: 
            ewa_loss = metrics['train_criterion_ewa'][-1]
        except IndexError:
            ewa_loss = 0
        
        losses,ewa_losses = train_one_epoch(model,opt,lr_scheduler,loss_fn,train_dataloader,args,transform,writer,ewa_loss)
        metrics['train_criterion'] = np.concatenate( (metrics['train_criterion'],losses.detach().cpu().numpy()) )
        metrics['train_criterion_ewa'] = np.concatenate( (metrics['train_criterion_ewa'],ewa_losses.detach().cpu().numpy()) )
                            
        # Compute validation metrics if validation period 
        args.current_count += 1 
        


        # Print log if log period
        if ( args.current_count % args.log_period == args.log_period - 1) or (args.current_count == (args.epochs-1) ):

            inner_compute_metrics(model,metrics,writer,args.current_count)

            strtoprint =f"epoch: {str(args.current_count)} " + \
                        f"train loss: {metrics['train_loss'][-1]:.4f} " + \
                        f"train acc: {metrics['train_acc'][-1]*100:2.4f} " + \
                        f"val loss: {metrics['val_loss'][-1]:.4f} " + \
                        f"val acc: {metrics['val_acc'][-1]*100:2.4f} " + \
                        f"test loss: {metrics['test_loss'][-1]:.4f} " + \
                        f"test acc: {metrics['test_acc'][-1]*100:2.4f} " 
                        
            if args.debug:
                # get current lr
                try: 
                    last_lr = opt.param_groups[0]['lr'][-1] if lr_scheduler is None else lr_scheduler.get_last_lr()[-1]
                except TypeError:
                    last_lr = opt.param_groups[0]['lr'] if lr_scheduler is None else lr_scheduler.get_last_lr()
                strtoprint += f"LR{last_lr:2.8f}"
                          
                                
            print(strtoprint, file=open(args.logpath, 'a'), flush=True)
            print(strtoprint)

        # Save checkpoint if save_period or best so far
        current_val_acc = metrics['test_acc'][-1]
        if args.save_period and ((args.current_count % args.save_period == args.save_period - 1) or (args.current_count == (args.epochs-1))) or current_val_acc > best_val_acc:

            m_path = args.modelpath + f"-e{str(args.current_count)}" + f"-a{current_val_acc*100:2.0f}" + '-m.pt'
            print(f'# Saving Model : {m_path}', file=open(args.logpath, 'a'), flush=True)
            utils.save_checkpoint(args.current_count,metrics,model,opt,args,m_path)

            best_val_acc = current_val_acc

            
            # Make visualization if debug and 
            model_to_vizulize = model
            if args.debug and isinstance(model_to_vizulize,Unet):
                viz_folder = os.path.join(args.out,'figs')
                visulize_batch(model_to_vizulize,train_dataloader,writer=writer,viz_folder=viz_folder,id_str=f"trn-e{args.current_count}-a{current_val_acc*100:2.0f}")
                if validation_dataloader is not None:
                    visulize_batch(model_to_vizulize,validation_dataloader,writer=writer,viz_folder=viz_folder,id_str=f"val-e{args.current_count}-a{current_val_acc*100:2.0f}")
                
                visulize_batch(model_to_vizulize,test_dataloader,writer=writer,viz_folder=viz_folder,id_str=f"tst-e{args.current_count}-a{current_val_acc*100:2.0f}")
                  

               
    return metrics


def train_one_epoch(m:nn.Module,opt,lr_scheduler,loss_fn:Callable,dl,args,augumentation=None,writer=None,ewa_loss=0)->Tuple[torch.Tensor,torch.Tensor]:
    """
    The functions trains the neural network model 'm' for one epoch of dataloader 'dl' using the provided optimizer 'opt' 
    and loss function 'loss_fn'.

    :param m: a PyTorch module (a neural network model)
    :param opt: a Pytorch Optimizer
    :param lr_scheduler: a Pytorch scheduler 
    :param loss_fn: a loss function that takes two inputs and returns a value or values of the loss between the two inputs
    :param dl: a PyTorch DataLoader that yields mini-batches of features and targets for evaluation
    :param args: an EasyDict: A dictionary-like object that contains the training hyperparameters (device,ewa_coef) 
    :parapm augumentation (Optional[nn.Module]): an augumentation to be applied to features before forward pass. Default None
    :param ewa_loss: last recorded ewa loss of training 


    :returns: a Tuple of losses and ewa_losses computed for each batch of dl. (torch.Tensors) 
    """
    #
    opt.zero_grad()
    m = m.to(args.device)
    m.train()

    losses = torch.zeros(len(dl))
    ewa_losses = torch.zeros(len(dl))

    bar = Bar('Training', max=len(dl))
    for idx,(features,targets) in enumerate(dl):
        
        features = features.to(args.device)

        if augumentation is not None:
            features,targets = utils.apply_transformation(augumentation,features,targets)
            
        targets = targets.to(args.device)
        targets_hat = m.forward(features)

        targets = ensure_onehot(targets,num_classes=targets_hat.shape[1]).to(float)

        l = torch.sum(loss_fn(targets_hat,targets)) / (targets_hat.numel() / targets_hat.shape[1])
        l.backward()
        opt.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        # ewa
        if (idx == 0 and ewa_loss == 0):
            ewa_loss = l
        else:
            ewa_loss = args.loss_ewa_coef * ewa_loss + (1-args.loss_ewa_coef) * l

        ewa_losses[idx] = ewa_loss
        losses[idx] = l

        if writer is not None:
            writer.add_scalars('Loss',{ 'loss': l,
                                        'tst': ewa_loss},idx + args.current_count * len(dl))
            
        # get current lr
        try: 
            last_lr = opt.param_groups[0]['lr'][-1] if lr_scheduler is None else lr_scheduler.get_last_lr()[-1]
        except TypeError:
            last_lr = opt.param_groups[0]['lr'] if lr_scheduler is None else lr_scheduler.get_last_lr()

        bar.suffix  = f"#({idx}/{len(dl)})#({args.current_count}/{args.epochs})|{bar.elapsed_td}|ETA:{bar.eta_td}|"+ \
                f"L_ewa:{ewa_loss:.4f}|L:{l:.4f}|last LR: {last_lr:.10f}|"
        bar.next()

    bar.finish()


    return losses, ewa_losses 