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
from typing import Optional, Union

def train():
    pass


def mixmatch(labeled_batch:torch.Tensor,labels:torch.Tensor,unlabeled_batch:torch.Tensor,clf:nn.Module,
                augumentation:custom_transforms.MyAugmentation,K:int,T:float,alpha:float,
                eps:float=1e-4)->Tuple[torch.Tensor,torch.Tensor]:
    """MixMatch algorithm implementation of https://arxiv.org/abs/1905.02249
    
    Args: 
        labeled_batch (torch.Tensor):   feature batch [N,...] containing N features (coresponding to known labels)
        labels(torch.Tensor):           labels (in form of one hot vectors) [N,C,...]
        unlabeled_batch (torch.Tensor): feature batch [N,...] containing N unlabeled features
        clf (nn.Module):                classifier to predict label distribution
        augumentation (nn.Module):      augumentation appliable on features 
        K (int):                        number of applications of augumentation to unlabeled features
        T (float >0):                   hyperparameter for sharpening the distribution (T->0 forces q->dirac)
        alpha (float>0):                hyperparameter of Beta distribution 
        eps (float>0):                  epsilon to detremine if element in the input is valid. (Sum across dim 1 
        (colors in images) should be at least eps)

    Returns:
        tuple(alternated_labeled_batch,alternated_unlabeled_batch), 
    """
     # change later 
    # 1) Augumentation and Data Guessing

    # 1a) One augumentation for each image in labeled batch

    if labels.ndim == labeled_batch.ndim and labels.shape[-2:] == labeled_batch.shape[-2:]:
        # Segmentation -> Do transformation on both images and labels
        aug_labeled_batch,aug_labels = augumentation(labeled_batch,labels)
        
    else:
        # Classification -> Do transformation only on images
        aug_labeled_batch,_ = augumentation(labeled_batch,None)
        aug_labels = labels 
    
    
    
    X_hat = (aug_labeled_batch,aug_labels)
    
    # 1b) K augumentations for each image in unlabeled batch

    N,C,H,W = unlabeled_batch.shape
    batch_repeated = unlabeled_batch.repeat(1, K, 1, 1).reshape(N,K,C,H,W)
    #aug_labeled_batch = augumentation(unlabeled_batch)
     # apply iteratively if augumentation does not support different augmentation for each image
    aug_labeled_batch = []
    for i in range(K):
        ith_augmented_batch,_ = augumentation(batch_repeated[:,i],None)
        aug_labeled_batch.append(ith_augmented_batch.unsqueeze(1))
    aug_unlabeled_batch = torch.cat(aug_labeled_batch,dim=1).reshape(N*K,C,H,W)

    predictions = nn.functional.softmax(clf(aug_unlabeled_batch),dim=1)    
    
    if predictions.ndim == aug_unlabeled_batch.ndim and predictions.shape[-2] == aug_unlabeled_batch.shape[-2]:
        # Segmentation -> Validate predictions, invert them,then average them and transform them once again
        validity_mask = (torch.sum(aug_unlabeled_batch,dim=1,keepdim=True) > eps).to(torch.float32)
        predictions = predictions * validity_mask

        base_predictions = augumentation.inverse_last_transformation(predictions)
        base_averages = average_labels(base_predictions.reshape(N,K,C,H,W),keepshape=True).reshape(N*K,C,H,W)
        _,avg_predictions = augumentation.apply_last_transformation(mask=base_averages) 

    else:
        # Classification -> Simple average
        avg_predictions = average_labels(predictions.reshape(N,K, *predictions.shape[1:]),
                                         keepshape=True
                                         ).reshape(N*K,*predictions.shape[1:])
    

    # 2) Sharpen distribution with temperature T
    sharp_avg_predictions = sharpen(avg_predictions,T=T,dim=1)
    U_hat = (aug_unlabeled_batch,sharp_avg_predictions)
    
    # 3) Concat and Shuffle 
    W = concatenate_and_shuffle(X_hat,U_hat)
    
    # split W into two
    W_1 = (W[0][:N],W[1][:N])
    W_2 = (W[0][N:],W[1][N:])
    
    # 4) Mix Up
    X_prime, x_lam = mixup(X_hat,W_1,alpha=alpha,eps=eps)
    U_prime, u_lam = mixup(U_hat,W_2,alpha=alpha,eps=eps)

    return X_prime, U_prime


def mixup(A:Tuple[torch.Tensor,torch.Tensor], B:Tuple[torch.Tensor,torch.Tensor], alpha=1.0,eps=1e-3,lam:Optional[torch.Tensor] = None):
    """
    Mixes up the input data and labels using the MixUp method.

    Args:
        A (tuple): Input data and label tensor tuple of shape (N, ...) and (N,C,), respectively.
        B (tuple): Input data and label tensor tuple of shape (N, ...) and (N,C,), respectively.
        alpha (float): MixUp hyperparameter.
        device (torch.device): Device to use (default: 'cpu').
        eps (float): Epsilon to detremine if element in the input is valid.
            (Sum across dim 1 (colors in images) should be at least eps)
        lam (Optional[torch.Tensor]): Lambda (mixing coeficient) [N], mainly for debugging purposes

        Remark: As we use  lam = max(lam,1-lam), the ordering matters! Do (X,W) or (U,W) not otherwise 

    Returns:
        A tuple of tuples (mixed_x,mixed_y),lam
          mixed-up data and labels, and the mixing coefficients:
        - mixed_x: Mixed-up data tensor.
        - mixed_y: Mixed-up label tensor.
    """
    # Unpack the input tuples
    x1, y1 = A
    x2, y2 = B

    # Compute mixing coefficients
    N,C = x1.shape[:2]
    if lam is None:
        lam = torch.Tensor(np.random.beta(alpha, alpha, N))
    
    lam = torch.max(lam,1-lam).to(x1.device)
    lams = torch.ones(x1.shape).to(x1.device) * lam.view(N,*( [1]*(x1.ndim-1)))
        


    # Due to the affine transformations, part of the image can be empty:
    # TODO QESTION: If view in seond pic is zero (due to transformation), do not apply weightening (Should it be important?)
    
    # If only x1 valid -> alpha = 1, if only x2 valid -> alpha = 0 
    # x1_mask = (x1.sum(axis=1,keepdim=True) > eps).repeat(1,C,*( [1] * (x1.ndim -2) ) )
    # x2_mask = (x2.sum(axis=1,keepdim=True) > eps).repeat(1,C,*( [1] * (x1.ndim -2) ) )
    # lams[torch.logical_and(x1_mask, torch.logical_not(x2_mask))] = 1
    # lams[torch.logical_and(x2_mask, torch.logical_not(x1_mask))] = 0

    # Mix up the input data and labels
    
    mixed_x = lams * x1 + (1 - lams) * x2
    
    if x1.ndim == y1.ndim and x1.shape[-2:] == y1.shape[-2:]: 
        # Segmetation -> mix with same lams as on image (each pixel)
        _lams = lams
    else:
        # Classification -> mix with lams for each sample (do not use image shapes logic)
        _lams = torch.ones(y1.shape).to(y1.device) * lam.view(N, *([1]* (y1.ndim-1)))
    
    mixed_y = _lams * y1 + (1 - _lams) * y2
        
    return (mixed_x, mixed_y),lam


def sharpen(p, T, dim=1):
    """Generated by GPT-3"""

    # Ensure that T is a positive number
    assert T > 0, "Temperature T must be a positive number."

    # Calculate sharpened probabilities
    p_sharp = p.pow(1 / T)
    p_sharp = torch.div(p_sharp, p_sharp.sum(dim=dim, keepdim=True))

    # Return sharpened probabilities
    return p_sharp


def concatenate_and_shuffle(X_hat:tuple[torch.Tensor,torch.Tensor], U_hat:tuple[torch.Tensor,torch.Tensor],shuffle_indices:Optional[torch.Tensor]=None):
    """
    Concatenates and shuffles two PyTorch tuples `X_hat` and `U_hat`.

    Args:
        X_hat (tuple): Tuple of two PyTorch tensors of shapes (N, ...) and (N,), representing input data and labels.
        U_hat (tuple): Tuple of two PyTorch tensors of shapes (N, ...) and (N,), representing input data and labels.
        Data and labels should have same size (across datasets X_hat,U_hat)
        shuffle_indicies Optional(torch.Tensor): Tensor of permuting indicies for debugging purposes

    Returns:
        Tuple of two PyTorch tensors of shuffled data and labels W:
        - (data_shuffled, label_shuffled): Shuffled tuples of data and labels.
        
    """
    # Concatenate X_hat and U_hat components
    data = torch.cat([X_hat[0],U_hat[0]], dim=0)
    labels = torch.cat([X_hat[1],U_hat[1]], dim=0)
    
    return shuffle(data,labels,shuffle_indices=shuffle_indices)


def shuffle(data:torch.Tensor, labels:torch.Tensor,shuffle_indices:Optional[torch.Tensor]=None):
    """
    Shuffles two PyTorch tensors `data` and `labels` with the same permutation.
    
    Args:
        data (torch.Tensor): Input tensor of shape (N, ...).
        labels (torch.Tensor): Tensor of labels of shape (N,).
        shuffle_indicies Optional(torch.Tensor): Tensor of permuting indicies for debugging purposes.

    Returns:
        Tuple of two torch.Tensors:
        - data_shuffled: Shuffled tensor with the same shape as `data`.
        - labels_shuffled: Shuffled tensor of labels with the same shape as `labels`. Mainly for debugging purposes
    """
    
    assert data.shape[0] == labels.shape[0], "Data and labels must have the same number of samples."
    
    # Generate a random permutation of the indices of the tensors
    indices = torch.randperm(data.size(0)) if shuffle_indices is None else shuffle_indices.long()

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
            eps(float): epsilon to determine valid elements of segmentation (sum across C should yield at least 1-eps)
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



# MIT License

# Copyright (c) 2019 Qing Yu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

# print(f"{len(xy)=}")
# print(f"{xy[0].shape=}")
# print(f"{nu=}")
# print(f"{offsets=}")
# print(f"{len(xy)=}")
# print(f"{len(xy[0])}")
# print(f"{xy[0][0].shape=}")