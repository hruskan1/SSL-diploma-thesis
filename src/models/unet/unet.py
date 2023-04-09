""" unet.py Module containing architecture of Unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia as K
from kornia.geometry.transform import Rescale,Resize
from typing import Callable, Tuple, List
import os
from tqdm import tqdm_notebook as tqdm
from time import time
import torchvision as tv
from easydict import EasyDict
import matplotlib.pyplot as plt
from typing import Optional,List,Union
from ...mixmatch import utils
from progress.bar import Bar as Bar

class ConvBlock(nn.Module):
    """ConvBlock"""
    def __init__(self,args):
        super().__init__()

        niters=args.niters
        in_ch = args.in_ch
        mid_ch = args.mid_ch
        out_ch = args.out_ch
        k = args.conv_kernel
        s = args.conv_stride
        p = args.conv_paddings
        g = args.get('group_size',None)

        feats = [nn.Conv2d(in_ch, mid_ch, kernel_size=k,stride=s,padding=p),
                _select_batch(args,mid_ch,g),
                nn.ReLU(inplace=True)]
        
        for i in range(0,niters-2,1):
            feats += [nn.Conv2d(mid_ch, mid_ch, kernel_size=k,stride=s,padding=p),
                    _select_batch(args,mid_ch,g),
                    nn.ReLU(inplace=True)]
        
        feats += [nn.Conv2d(mid_ch, out_ch, kernel_size=k,stride=s,padding=p), 
                _select_batch(args,out_ch,g),
                nn.ReLU(inplace=True)]

        self.features = nn.Sequential(*feats)
        
    def forward(self, x):
        return self.features(x)


class EncoderBlock(nn.Module):
    """EncoderBlock"""
    def __init__(self,args):
        super().__init__()

        self.conv_layer = ConvBlock(args)
        if args.use_kornia == True:
            self.scaling_layer = Rescale(factor = args.scale,
                                    interpolation=args.interpolation,
                                    antialias=True)
        else:
            raise NotImplementedError("We have not implemented maxpooling!")
        
    def forward(self,x):
        skip_x = self.conv_layer(x)
        x = self.scaling_layer(skip_x)
        return x,skip_x


class DecoderBlock(nn.Module):
    """Decoder Block"""
    def __init__(self,args):
        super().__init__()

        if args.use_kornia == True:
            self.scaling_layer = Rescale(factor = args.scale,
                                    interpolation = args.interpolation,
                                    antialias = True)
        else:
            raise NotImplementedError("We have not implemented maxpooling!")
        
        self.conv_layer = ConvBlock(args)
        

    def forward(self,x,skip_x):
        
        x = self.scaling_layer(x)
        x = torch.cat((skip_x,x),dim=1) # concat skip 
        return self.conv_layer(x)


class Bottleneck(nn.Module):
    """Bottleneck module"""
    def __init__(self,args):
        super().__init__()

        self.body =  ConvBlock(args)
        
    def forward(self,x):
        return self.body(x)


def _select_batch(block_args,channels,group_size):
    
    if block_args.normalization_type == 'batch':
        return nn.BatchNorm2d(num_features=channels)

    elif block_args.normalization_type == 'group':
        return(nn.GroupNorm(int(max(1,channels/group_size)), channels))


class Unet(nn.Module):
    """Creates U-shaped net with Encoder and Decoder part specified by EasyDict args"""
    def __init__(self,args:EasyDict):
        super().__init__()
        encoders = []
        decoders = []
        for block_index, block_parameters in enumerate(args.blocks):
            
            if block_parameters.type == "encoder":
                encoders.append(EncoderBlock(block_parameters))
        
            elif block_parameters.type == "decoder":
                decoders.append(DecoderBlock(block_parameters))
            
            elif block_parameters.type == "bottleneck":
                bottleneck = Bottleneck(block_parameters)

            elif block_parameters.type == "classifier":
                b = block_parameters
                clf = nn.Conv2d(b.in_ch,b.out_ch,kernel_size = b.conv_kernel)

            else:
                raise ValueError(f"Type not recognized: {block_parameters.type}!")
    
        self.encoders = nn.ModuleList(encoders)
        self.bottleneck = bottleneck
        self.decoders = nn.ModuleList(decoders)
        self.clf = clf

        self.apply(_weight_init)

    def forward(self, input):
        
        x = input
        skip_outputs = []
            
        for i,l in enumerate(self.encoders):
            x, skip = l(x)
            skip_outputs.append(skip)
        
        x = self.bottleneck(x)
        
        for i,l in enumerate(self.decoders):
            x = l(x,skip_outputs[-i-1])

        output = self.clf(x)
        return output


def _weight_init(m) -> None:
    '''Function, which fills-in weights and biases for layers'''

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    
        if m.bias is not None:
            nn.init.uniform_(m.bias)

# TODO: NOT YET REIMPLEMENTED SOMEWHERE ELSE:
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



def make_predictions(m:nn.Module,args:EasyDict,dataset:torch.utils.data.Dataset,index_list:Optional[List[int]])->List[torch.Tensor]:
    """
    Predict a dataset and return a list 
    """
    if index_list is None:
        index_list = range(len(dataset))
    inputs = []
    targets = []
    outputs = []
    m = m.eval()
    with torch.no_grad():
        for index in index_list:
            input,target = dataset[index]
            input = input.unsqueeze(0).to(args.device)
            target = target.unsqueeze(0).to(args.device)

            inputs.append(input)
            targets.append(target)
            outputs.append(m.forward(input))
    
    inputs = torch.cat(inputs).cpu()
    targets = torch.cat(targets).cpu()
    outputs = torch.cat(outputs).cpu()
    return [inputs,targets,outputs]



