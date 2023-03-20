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
import numpy as np

def train():
    pass



def mixmatch(labeled_batch:torch.Tensor,labels:torch.Tensor,unlabeled_batch:torch.Tensor,clf:nn.Module,
            augumentation:custom_transforms.MyAugmentation,K:int,T:float,alpha:float,device:torch.device=torch.device('cpu'))->Tuple[torch.Tensor,torch.Tensor]:
    """MixMatch algorithm implementation of https://arxiv.org/abs/1905.02249
    
    Args: 
        labeled_batch (torch.Tensor): feature batch [N,...] containing N features (coresponding to known labels)
        labels(torch.Tensor): labels (in form of one hot vectors) [N,C,...]
        unlabeled_batch (torch.Tensor): feature batch [N,...] containing N unlabeled features
        clf (nn.Module): classifier to predict label distribution
        augumentation (nn.Module): augumentation appliable on features 
        K (int): number of applications of augumentation to unlabeled features
        T (float >0): hyperparameter for sharpening the distribution (T->0 forces q->dirac)
        alpha (float>0) : hyperparameter of Beta distribution 

    Returns:
        tuple(alternated_labeled_batch,alternated_unlabeled_batch), 
    """
    eps=10**(-4) # change later 


    # 1) Augumentation and Data Guessing

    # 1a) One augumentation for each image in labeled batch

    if labels.shape == labeled_batch.shape:
        # Segmentation -> Do transformation on both images and labels
        aug_labeled_batch,aug_labels = augumentation(labeled_batch,labels)
        
    else:
        # Classification -> Do transformation only on images
        aug_labeled_batch,_ = augumentation(labeled_batch,None)
        aug_labels = labels 
    
    X_hat = (aug_labeled_batch,aug_labels)
    
    # 1b) K augumentations for each image in unlabeled batch

    N,C,H,W = unlabeled_batch.shape
    batch_repeated = unlabeled_batch.repeat(1, K, 1, 1).reshape(N*K,C,H,W)
    aug_unlabeled_batch,_ = augumentation(batch_repeated,None) 

    ### until we do not have clf
    _l = labels.repeat(1, K, 1, 1).reshape(N*K,C,H,W)
    _,predictions = augumentation.apply_last_transformation(None,_l)
    # aug_predictions = clf(aug_unlabeled_batch) 
    ### 
    
    validity_mask = (torch.sum(aug_unlabeled_batch,dim=1,keepdim=True) > eps).to(torch.float32)
    predictions = predictions * validity_mask

    if aug_labels.shape[-2] == aug_unlabeled_batch.shape[-2]:
        # Segmentation -> Invert the labels, average them and transform them once again
        base_predictions = augumentation.inverse_last_transformation(predictions)
        base_averages = average_labels(base_predictions.reshape(N,K,C,H,W),keepshape=True).reshape(N*K,C,H,W)
        _,avg_predictions = augumentation.apply_last_transformation(mask=base_averages) 

    else:
        # Classification -> Simple average
        avg_predictions = average_labels(predictions.reshape(N,K, *predictions.shape[1:])).reshape(N*K,*predictions.shape[1:])
    

    # 2) Sharpen distribution with temperature T
    sharp_avg_predictions = sharpen(avg_predictions,T=T,dim=1)

    U_hat = (aug_unlabeled_batch,sharp_avg_predictions)
    
    # 3) Concat and Shuffle 
    W = concatenate_and_shuffle(X_hat,U_hat)

    print(f"{X_hat[0].shape=},{X_hat[1].shape=}")
    print(f"{U_hat[0].shape=},{U_hat[1].shape=}")
    print(f"{W[0].shape},{W[0].shape}")
    
    # split W into two
    W_1 = (W[0][:N],W[1][:N])
    W_2 = (W[0][N:],W[1][N:])

    print(f"{W_1[0].shape},{W_1[0].shape}")
    print(f"{W_2[0].shape},{W_2[0].shape}")

    X_prime = mixup(X_hat,W_1,alpha=alpha,device=device)
    U_prime = mixup(U_hat,W_2,alpha=alpha,device=device)


    utils.plot_batch(torch.cat(X_hat,dim=0))
    utils.plot_batch(torch.cat(W_1,dim=0))
    utils.plot_batch(torch.cat(X_prime,dim=0))

    utils.plot_batch(torch.cat(U_hat,dim=0))
    utils.plot_batch(torch.cat(W_2,dim=0))
    utils.plot_batch(torch.cat(U_prime,dim=0))

    return X_prime, U_prime
    
    #inverse_unlabeled_batch = augumentation.inverse_last_transformation(aug_unlabeled_batch) # for debug
    # print("Valid region masks")
    # utils.plot_batch(valid_region_mask)
    # print(f"aug_unlabeled_batch,{aug_unlabeled_batch.shape}")
    # utils.plot_batch(aug_unlabeled_batch)
    # print(f"aug_predictions,{aug_predictions.shape}")
    # utils.plot_batch(aug_predictions)
    # print(f"inverse_aug_predictions,{inverse_aug_predictions.shape}")
    # utils.plot_batch(inverse_aug_predictions)
    # print(f"average_inverse_aug_predictions,{avreage_inverse_aug_predictions.shape}")
    # utils.plot_batch(avreage_inverse_aug_predictions)
    # print(f"final_predictions,{avg_predictions.shape}")
    # utils.plot_batch(avg_predictions)
    # print(f"inverse_unlabeled_batch, {inverse_unlabeled_batch.shape}")
    # utils.plot_batch(inverse_unlabeled_batch)
    # print(f"first img in batch:")
    # utils.plot_batch(inverse_unlabeled_batch.reshape(K,N,C,H,W)[0].reshape(N,C,H,W))

    # Compute average





def mixup(A:Tuple[torch.Tensor,torch.Tensor], B:Tuple[torch.Tensor,torch.Tensor], alpha=1.0, device:torch.device=torch.device('cpu')):
    """
    Mixes up the input data and labels using the MixUp method.

    Args:
        A (tuple): Input data and label tensor tuple of shape (N, ...) and (N,C,), respectively.
        B (tuple): Input data and label tensor tuple of shape (N, ...) and (N,C,), respectively.
        alpha (float): MixUp hyperparameter.
        device (torch.device): Device to use (default: 'cpu').

        Remark: As we use  lam = max(lam,1-lam), the ordering matters! Do (X,W) or (U,W) not otherwise 

    Returns:
        A tuple of mixed-up data and labels, and the mixing coefficients:
        - mixed_x: Mixed-up data tensor.
        - mixed_y: Mixed-up label tensor.
    """
    # Unpack the input tuples
    x1, y1 = A
    x2, y2 = B

    # Compute mixing coefficients
    N = x1.shape[0]
    lam = torch.Tensor(np.random.beta(alpha, alpha, N)).to(device)
    lam = torch.max(lam,1-lam) 

    # Mix up the input data and labels

    # TODO: If segmentation (decide if last two shapes of data and labels are same?)
    #       If view in seond pic is zero (due to transformation), do not apply weightening (Should it be important?)
    mixed_x = lam.view(N, *( [1]*(x1.ndim-1)) ) * x1 + (1 - lam.view(N,*( [1]*(x2.ndim-1))) ) * x2
    mixed_y = lam.view(N, *( [1]*(y1.ndim-1)) ) * y1 + (1 - lam.view(N, *([1]*(y2.ndim-1)) )) * y2

    return mixed_x, mixed_y


def sharpen(p, T, dim=1):
    """Generated by GPT-3"""

    # Ensure that T is a positive number
    assert T > 0, "Temperature T must be a positive number."

    # Calculate sharpened probabilities
    p_sharp = p.pow(1 / T)
    p_sharp = torch.div(p_sharp, p_sharp.sum(dim=dim, keepdim=True))

    # Return sharpened probabilities
    return p_sharp


def concatenate_and_shuffle(X_hat:tuple[torch.Tensor,torch.Tensor], U_hat:tuple[torch.Tensor,torch.Tensor]):
    """
    Concatenates and shuffles two PyTorch tuples `X_hat` and `U_hat`.

    Args:
        X_hat (tuple): Tuple of two PyTorch tensors of shapes (N, ...) and (N,), representing input data and labels.
        U_hat (tuple): Tuple of two PyTorch tensors of shapes (N, ...) and (N,), representing input data and labels.
        Data and labels should have same size (across datasets X_hat,U_hat)

    Returns:
        Tuple of two PyTorch tensors of shuffled data and labels W:
        - (data_shuffled, label_shuffled): Shuffled tuples of data and labels.
        
    """
    # Concatenate X_hat and U_hat components
    data = torch.cat([X_hat[0],U_hat[0]], dim=0)
    labels = torch.cat([X_hat[1],U_hat[1]], dim=0)

    return shuffle(data,labels)


def shuffle(data:torch.Tensor, labels:torch.Tensor):
    """
    Shuffles two PyTorch tensors `data` and `labels` with the same permutation.
    
    Args:
        data (torch.Tensor): Input tensor of shape (N, ...).
        labels (torch.Tensor): Tensor of labels of shape (N,).

    Returns:
        Tuple of two torch.Tensors:
        - data_shuffled: Shuffled tensor with the same shape as `data`.
        - labels_shuffled: Shuffled tensor of labels with the same shape as `labels`.
    """
    
    assert data.shape[0] == labels.shape[0], "Data and labels must have the same number of samples."

    # Generate a random permutation of the indices of the tensors
    indices = torch.randperm(data.size(0))

    # Apply the permutation to both tensors
    data_shuffled = data[indices]
    labels_shuffled = labels[indices]

    # Return the shuffled tensors
    return data_shuffled, labels_shuffled


def average_labels(labels:torch.Tensor,keepshape=False,eps=0.5)->torch.Tensor:
    """Augument the batches and provide the labels for unlabeled_batch
    
       Args:
            labeled_batch (torch.Tensor): feature batch [N,K,C,..],
                where K are number of label instances for each image, N is number of images in batch
                and C is number of segmentation classes
            keepshape (bool): See 'Returns'
            eps(float): epsilon to determine valid elements of segmentation (sum across C should yield 1-eps)
        Returns:
            averaged batch [N,C,...] Avreage is computed only from those labels, which are valid 
            if keepshape:
                averaged batch [N,K,C,..], where average is populated across first dim  
    """
    N,K = labels.shape[:2]

    label_sums = labels.sum(dim=1,keepdim=True)
    valid_elements = (labels.sum(dim=2,keepdim=True) >= 1-eps).to(torch.float32)
    ns = valid_elements.sum(dim=1,keepdim=True) 
    avgs = label_sums / ns

    if keepshape: 
        avgs = avgs.repeat(1,K,*([1] * (labels.ndim-2))).reshape(N,K,*labels.shape[2:])
    else:
        avgs = avgs[:,0]

    return avgs
