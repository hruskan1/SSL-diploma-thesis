# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# Copyright is valid only for first three losses 

"""Custom loss functions"""

import torch
from torch.nn import functional as F


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes

# Not under original license:
def soft_cross_entropy(input, target,weight: torch.Tensor = None,reduction:str = 'none'):
    """
    Computes soft-CrossEntropy loss (inspired from pytorch code)
    """
    assert input.shape == target.shape

    logprobs = torch.nn.functional.log_softmax(input, dim = 1)
    
    if weight is None:
            weight = 1
    if reduction == 'none':
        return  -(weight * target * logprobs)
    elif reduction == 'sum':
        return  -(weight * target * logprobs).sum()
    elif reduction == 'mean':
        return  -(weight * target * logprobs).sum() / ( torch.numel(input) /input.size(dim=1) ) # all other dim


def mse_softmax(input,target,reduction='none'):
    """
    Computes mse of softmax of input with target 
    """
    assert input.shape == target.shape

    return F.mse_loss(F.softmax(input,dim=1),target,reduction=reduction)
