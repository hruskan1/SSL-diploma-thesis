"""misc.py Everything used with NN models"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision as tv
from typing import Callable,Optional,Tuple
from progress.bar import Bar
import warnings
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from ..mixmatch.utils import apply_transformation,plot_batch
from ..mixmatch.datasets import CityScapeDataset
from .unet.unet import Unet
from .wide_resnet.wide_resnet import WideResNet
import matplotlib.pyplot as plt

def evaluate(m:nn.Module,loss_fn:Callable,dl,device:torch.device=torch.device('cpu')):
    """
    The function evaluates the neural network model m on the data in the data loader dl using the provided loss function loss_fn. 
    The evaluation is performed with torch.no_grad() to disable gradient computation and speed up computation. 
    The final returned value is the average loss per data item (picture).
    
    :param m: a PyTorch module (a neural network model)
    :param loss_fn: a loss function that takes two inputs and returns a value or values of the loss between the two inputs
    :param dl: a PyTorch DataLoader that yields mini-batches of features and targets for evaluation
    :param device: a pytorch device instance which specifies the gpu number or cpu device on which the evalutaion is running. 
    
    :returns: Tuple of (avreage loss,accuracy) (Average is compute across items in dataloder dl)
    """
    m = m.to(device)
    m.eval()
    loss,numel,accuracy = (0,0,0)

    bar = Bar('Evaluating', max=len(dl))
    with torch.no_grad():
      
        for idx,(features,targets) in enumerate(dl):
            targets = targets.to(device).to(torch.float32)
            features = features.to(device)
            targets_hat = m.forward(features) # [N,C,..]

            
            targets = ensure_onehot(targets,num_classes=targets_hat.shape[1])
            
                
            loss += torch.sum(loss_fn(targets_hat,targets))

            acc_mask = torch.argmax(targets_hat,dim=1) == torch.argmax(targets,dim=1)
            accuracy += torch.sum(acc_mask)
            numel += acc_mask.numel()

            bar.suffix  = f"#({idx}/{len(dl)})|{bar.elapsed_td}|ETA:{bar.eta_td}|L: {loss/numel:.4f}|acc:{(accuracy/numel)*100:2.4f}"
            bar.next()

    bar.finish()

    return loss / numel, accuracy / numel


def ensure_onehot(targets:torch.Tensor,num_classes):
    """Ensures that targets are one-hot encoded"""
    if targets.ndim == 1 or targets.shape[1] == 1:
                # [N,] or [N,1,..] -> [N,C,..]
                N = targets.shape[0]
                targets = F.one_hot(targets.reshape(N,*(targets.shape[2:])).to(torch.long),num_classes=num_classes).permute(0,3,1,2).to(torch.float32)
    
    return targets

def evaluate_IoU(model:nn.Module, dataloader:data.DataLoader)->Tuple[float,torch.Tensor]:
    """
    Computes the average IoU for each class over the entire dataset.

    :param model: A PyTorch model.
    :param dataloader: A PyTorch dataloader.
    :return: Tuple(average_iou,class_iou), float and tensor containg IoU on dataset for each class [C]
    """
    model.eval()
    device = next(model.parameters()).device

    # get output from model and through that obtain number of classes
    num_classes = model(next(iter(dataloader))[0].to(device)).shape[1] 

    # Initialize counters for each class
    intersection_total = torch.zeros(num_classes, dtype=torch.float32, device=device)
    union_total = torch.zeros(num_classes, dtype=torch.float32, device=device)

    bar = Bar('Evaluating IoU', max=len(dataloader))

    with torch.no_grad():
        for idx,(inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = ensure_onehot(targets,num_classes)

            # Compute predictions
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            onehot_preds = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()


            # Update the total intersection and union for each class
            current_intersection = torch.sum(onehot_preds * targets, dim=(0, 2, 3))
            intersection_total += current_intersection
            union_total += torch.sum(onehot_preds + targets, dim=(0, 2, 3)) - current_intersection
            
            _avg_iou = torch.mean( (intersection_total + 1e-15) / (union_total + 1e-15) )
            bar.suffix  = f"#({idx}/{len(dataloader)})|{bar.elapsed_td}|ETA:{bar.eta_td}|avg iou:{_avg_iou*100:2.5f}"
            bar.next()

    bar.finish()
            

    # Compute the average IoU score for each class
    class_iou = (intersection_total + 1e-15) / (union_total + 1e-15)
    average_iou = np.mean(class_iou.cpu().numpy())

    # Return the results as a dictionary
    return average_iou,class_iou

    

def IoU(pred, target,smooth=1):
    """
    Computes IoU (Jaccardi index) for multiclass prediction over the batched prediction and target

    :param pred:    torch.Tensor [N,C,H,W] or [N,C,HW] containing the prediction from model
    :param target:  torch.Tensor [N,C,H,W] or [N,C,HW] contating the ground truth. Have to be one-hot!

    :return tuple(intersection,union), where both are torch.Tensor [N,C].
    """
    
    if target.ndim == 3:
        N,C,HW = target.shape
    elif target.ndim == 4:
        N,C,W,H = target.shape
        HW = W*H
        target = target.reshape(N,C,HW)
    else: 
        raise ValueError(f"Dimensions do not fit: {target.ndim} ({target.shape})")

    
    intersection = (pred * target).sum(-1).float() 
    union = pred.sum(-1).float() + target.sum(-1).float() - intersection
    IoU = (intersection + smooth) / (union + smooth)
    return IoU


def F1(pred,target,smooth=1):
    """
    Computes F1 metrics for multiclass prediction over the batched prediction and target

    :param pred:    torch.Tensor [N,C,H,W] or [N,C,HW] containing the prediction from model
    :param target:  torch.Tensor [N,C,H,W] or [N,C,HW] contating the ground truth. Have to be one-hot

    :param smooth:  int - smoother coeficient to avoid division by zero
    :return torch.Tensor [N] of F1 metricies for each example in batch 
    """
    if target.ndim == 3:
        N,C,HW = target.shape
    elif target.ndim == 4:
        N,C,W,H = target.shape
        HW = W*H
        target = target.reshape(N,C,HW)
    else: 
        raise ValueError(f"Dimension does not fit: {target.ndim} ({target.shape})")
    
    intersection = (pred * target).sum(-1).float()
    sum_of_areas = pred.sum(-1).float() + target.sum(-1).float()
    dice = (2 * intersection + smooth) / (sum_of_areas + smooth)
    return torch.mean(dice,axis=1) # sum along classes



def visulize_batch(model:torch.nn.Module,
                   dl:torch.utils.data.DataLoader,
                   augmentation:Optional[torch.nn.Module]=None,
                   writer:Optional[SummaryWriter]=None,
                   viz_folder:Optional[str]=None,
                   id_str:str = '',
                   id_int:str = 0):
    """
    Visualize the first batch from dataloader 'dl' on current model 'model' and store it if 'viz_folder' specified. If writer, write into a writer.
    
    Parameters:
        - model: The PyTorch model to use for inference.
        - dl: The PyTorch DataLoader object that generates the data.
        - augmentation (optional): The PyTorch transform module to apply to the batch inputs before passing them to the model.
        - writer (optional): The TensorBoard SummaryWriter object to use for logging the visualizations.
        - viz_folder (optional): The folder path to store the visualization images. The folder will be created if it does not exist.
        - id (optional): The id string of the batch and dataloader for logging and visualization purposes.
    """
    if isinstance(model,Unet):
        _visulize_segmentation_model(model,dl,augmentation,writer,viz_folder,id_str,id_int)
    elif isinstance(model,WideResNet):
        _visulize_classification_model(model,dl,augmentation,writer,viz_folder,id_str,id_int)
    else:
         warnings.warn(f"Model class not recognized: {type(model)}")


def _visulize_classification_model(model:torch.nn.Module,
                                dl:torch.utils.data.DataLoader,
                                augmentation:Optional[torch.nn.Module]=None,
                                writer:Optional[SummaryWriter]=None,
                                viz_folder:Optional[str]=None,
                                id_str:str='',
                                id_int:int=0):
     raise NotImplementedError
     
    
def _visulize_segmentation_model(model:torch.nn.Module,
                                dl:torch.utils.data.DataLoader,
                                augmentation:Optional[torch.nn.Module]=None,
                                writer:Optional[SummaryWriter]=None,
                                viz_folder:Optional[str]=None,
                                id_str:str='',
                                id_int:int=0):
     
    model.eval()
    
    inputs, targets = next(iter(dl))

    # Apply augmentation if specified
    if augmentation:
        inputs_aug,target_aug = apply_transformation(augmentation,inputs,targets)
        inputs = torch.cat([inputs, inputs_aug], dim=0)
        targets = torch.cat([targets,target_aug], dim=0)

    
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1,keepdim=True)


    # Try to color if CityScapeDataset
    if isinstance(dl.dataset,CityScapeDataset):
        targets = CityScapeDataset.color_segmentation(targets)
        preds = CityScapeDataset.color_segmentation(preds)

    #! Remove normalization of inputs
    inputs = CityScapeDataset.remove_normalization(inputs)
    images = tv.utils.make_grid(torch.cat([inputs, targets, preds], dim=0),nrow=inputs.shape[0])
    
    # Save the visualization to the writer object if specified
    if writer:
        writer.add_image(f'Batch {id_str} (inputs,targets,outputs)', images, id_int)

    # Save the visualization to a file in the visualization folder if specified
    if viz_folder:
        os.makedirs(viz_folder, exist_ok=True)
        save_image(images, os.path.join(viz_folder, f'batch_{id_str}.png'))
        
