"""mixmatch.py : Module containing the main loop (algorithm) of MixMatch (Pytorch reimplementation)

https://github.com/google-research/mixmatch/
https://arxiv.org/abs/1905.02249
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms 
from typing import Tuple
import matplotlib.pyplot as plt
import transformations as custom_transforms
import utils
import copy

def train():
    pass



def mixmatch(labeled_batch:torch.Tensor,labels:torch.Tensor,unlabeled_batch:torch.Tensor,clf:nn.Module,
            augumentation:custom_transforms.MyAugmentation,K:int,temperature:float,alpha:float)->Tuple[torch.Tensor,torch.Tensor]:
    """MixMatch algorithm implementation of https://arxiv.org/abs/1905.02249
    
    Args: 
        labeled_batch (torch.Tensor): feature batch [N,...] containing N features (coresponding to known labels)
        labels(torch.Tensor): labels (in form of one hot vectors) [N,C]
        unlabeled_batch (torch.Tensor): feature batch [N,...] containing N unlabeled features
        clf (nn.Module): classifier to predict label distribution
        augumentation (nn.Module): augumentation appliable on features 
        K (int): number of applications of augumentation to unlabeled features
        temperature (float >0): hyperparameter for sharpening the distribution (T->0 forces q->dirac)
        alpha (float>0) : hyperparameter of Beta distribution 

    Returns:
        tuple(alternated_labeled_batch,alternated_unlabeled_batch), 
    """
    eps=10**(-4) # change later 

    # One augumentation for each image in labeled batch

    utils.plot_batch(labeled_batch)

    if labels.shape == labeled_batch.shape:
        # Segmentation -> Do transformation on both images and labels
        aug_labeled_batch,aug_labels = augumentation(labeled_batch,labels)
        utils.plot_batch(aug_labeled_batch)
        utils.plot_batch(aug_labels)
    else:
        # Classification -> Do transformation only on images
        aug_labeled_batch,_ = augumentation(labeled_batch,None)
        aug_labels = labels 
    

    # K augumentations for each image in unlabeled batch

    N,C,H,W = unlabeled_batch.shape
    batch_repeated = unlabeled_batch.repeat(1, 2, 1, 1).reshape(2*N,C,H,W)
    aug_unlabeled_batch,_ = augumentation(batch_repeated,None)
    
    # Create valid region masks
    valid_region_mask = torch.sum(aug_unlabeled_batch,dim=1,keepdim=True) > eps
    print(valid_region_mask.shape,valid_region_mask.dtype)

    #aug_predictions = clf(aug_unlabeled_batch)
    aug_predictions = torch.rand(aug_unlabeled_batch.shape) * valid_region_mask # until we have clf

    if aug_labels.shape[-2] == aug_unlabeled_batch.shape[-2]:
        # Segmentation -> Do inverse transformation on labels
        aug_predictions = augumentation.inverse_last_transformation(aug_predictions)
        inverse_unlabeled_batch = augumentation.inverse_last_transformation(aug_unlabeled_batch) # for debug
    
    utils.plot_batch(valid_region_mask)
    utils.plot_batch(aug_unlabeled_batch)
    utils.plot_batch(aug_predictions)
    utils.plot_batch(inverse_unlabeled_batch)


    # Compute average




    


def augument_and_label(labeled_batch,labels,unlabeled_batch,clf,aug_l,aug_u):
    """Augument the batches and provide the labels for unlabeled_batch
    
       Args:
            labeled_batch (torch.Tensor): feature batch [N,...] containing N features (coresponding to known labels)
            labels(torch.Tensor): labels (in form of one vectors) [N,C]
            unlabeled_batch (torch.Tensor): feature batch [N,...] containing N unlabeled features
            clf (nn.Module): classifier to predict label distribution
            aug_l (nn.Module): transformations applied on labled batch
            aug_u (nn.Module): transformations applied on unlabeled batch

        Returns:
            agumented_labeled_batch,augumented_labels,augumented_unlabeled_batch,augumented_guessed_labels
    """

    # Augument the labeled batch








