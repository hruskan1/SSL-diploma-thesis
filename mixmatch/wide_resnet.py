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

"""Wide ResNet implementation from https://github.com/YU1ut/MixMatch-pytorch/"""
# The license is valid only for clasess

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

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
    
    with torch.no_grad():
      
        for idx,(features,targets) in enumerate(dl):
            
            targets = targets.to(device).to(torch.float32)
            features = features.to(device)
            targets_hat = m.forward(features) # [N,C]

            if targets.ndim == 1 or targets.shape[1] == 1:
                # [N,] or [N,1] -> [N,C]
                N = targets.shape[0]
                C = targets_hat.shape[1]
                targets = F.one_hot(targets.reshape(N),num_classes=C)
            
            
            loss += torch.sum(loss_fn(targets_hat,targets))
            accuracy += torch.sum(torch.argmax(targets_hat,dim=1) == torch.argmax(targets,dim=1) )
            numel += targets_hat.shape[0]

    return loss / numel, accuracy / numel


def train_one_epoch(m:nn.Module,opt,loss_fn:Callable,dl,device:torch.device=torch.device('cpu')):
    """
    The functions trains the neural network model 'm' for one epoch of dataloader 'dl' using the provided optimizer 'opt' 
    and loss function 'loss_fn'.

    :param m: a PyTorch module (a neural network model)
    :param opt: a Pytorch Optimizer
    :param loss_fn: a loss function that takes two inputs and returns a value or values of the loss between the two inputs
    :param dl: a PyTorch DataLoader that yields mini-batches of features and targets for evaluation
    :param device: a pytorch device instance which specifies the gpu number or cpu device on which the evalutaion is running. 
    """
    #TODO: Write a standard (supervised) training loop
    pass
