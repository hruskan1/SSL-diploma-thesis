import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import OneHotCategorical
from torch.nn.functional import relu


class StochSign(nn.Module):
    """
    Stochastic Sign function with straight-through gradient estimator
    
    Shape:
        - Input: (N, *) where * means additional dimensions
        - Output: (N *), same as input
    """
    def __init__(self, tau=1.0):
        super(StochSign, self).__init__()
        self.tau = tau

    def forward(self, x):
        p0 = x.mul(self.tau).sigmoid()
        out = 2.0 * p0.bernoulli() - 1.0
        # take value of the sample and gradient of tanh = 2*sigmoid - 1
        out = out.detach() + 2 * (p0 - p0.detach())
        return out


class StochHeavi(nn.Module):
    """
    Stochastic Heaviside function with straight-through gradient estimator

    Shape:
        - Input: (N, *) where * means additional dimensions
        - Output: (N *), same as input
    """

    def __init__(self, tau=1.0):
        super(StochHeavi, self).__init__()
        self.tau = tau

    def forward(self, x):
        p0 = x.mul(self.tau).sigmoid()
        out = p0.bernoulli()
        # take value of the sample and gradient of sigmoid
        out = out.detach() + (p0 - p0.detach())
        return out


class StochArgmax(nn.Module):
    """
    Stochastic Argmax function with straight-through gradient estimator

    Shape:
        - Input: (N,dim)
        - Output: (N,dim), same as input
    """

    def __init__(self, tau=1.0):
        super(StochArgmax, self).__init__()
        self.tau = tau

    def forward(self, x):
        p0 = x.mul(self.tau).softmax(dim=-1)
        dd = OneHotCategorical(probs=p0)
        sample = dd.sample()

        return sample.detach() + (p0 - p0.detach())


class StochArgmax2d(nn.Module):
    """
    Stochastic Argmax function with straight-through gradient estimator

    Shape:
        - Input: (N,dim, *) where * means additional dimensions
        - Output: (N,dim, *), same as input
    """

    def __init__(self, tau=1.0):
        super(StochArgmax2d, self).__init__()
        self.tau = tau

    def forward(self, x):
        p0 = x.mul(self.tau).softmax(dim=1)
        dd = OneHotCategorical(probs=p0.transpose_())
        sample = dd.sample().transpose(1, -1)

        return sample.detach() + (p0 - p0.detach())


class StochReLU(nn.Module):
    """
    Stochastic ReLU

    Shape:
        - Input: (N, *) where * means additional dimensions
        - Output: (N *), same as input
    """

    def __init__(self, sigma=1.0):
        super(StochReLU, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        p0 = torch.randn_like(x).mul(self.sigma).add(1.0)
        out = F.relu(x.mul(p0))

        return out

def ZGR_categorical(logits, x=None):
    """Returns a categorical sample from Categorical softmax(logits) (over axis=-1) as a
    one-hot vector, with ZGR gradient.
    
    Input: 
    logits [*, C], where C is the number of categories
    x: (optional) categorical sample to use instead of drawing a new sample. [*]
    
    Output: categorical samples with ZGR gradient [*,C] encoded as one_hot
    """
    # return ZGR_Function().apply(logits, x)
    # using surrogate loss
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    p = logp.exp()  # [*, C]
    dx_ST = p  # [*,C]
    index = torch.distributions.categorical.Categorical(probs=p, validate_args=False).sample()  # [*]
    num_classes = logits.shape[-1]
    y = F.one_hot(index, num_classes=num_classes).to(p)  # [*,C], same dtype as p
    logpx = logp.gather(-1, index.unsqueeze(-1)) # [*,1] -- log probability of drawn sample
    dx_RE = (y - p.detach()) * logpx
    dx = (dx_ST + dx_RE) / 2
    return y + (dx - dx.detach())


def ZGR_binary(logits:Tensor, x:Tensor=None)->Tensor:
    """Returns a Bernoulli sample for given logits with ZGR = DARN(1/2) gradient
    Input: logits [*]
    x: (optional) binary sample to use instead of drawing a new sample. [*]
    Output: binary samples with ZGR gradient [*], dtype as logits
    """
    p = torch.sigmoid(logits)
    if x is None:
        x = p.bernoulli()
    J = (x * (1-p) + (1-x)*p )/2
    return x + J.detach()*(logits - logits.detach()) # value of x with J on backprop to logits


class ReSigm(nn.Module):
    """
    Mixed ReLU and Sigmoid

    Shape:
        - Input: (N, *) where * means additional dimensions
        - Output: (N *), same as input
    """

    def __init__(self, tau=1.0):
        super(ReSigm, self).__init__()

    def forward(self, x):
        hhr, hht = torch.chunk(x, 2, dim=1)
        return torch.cat((torch.relu(hhr), torch.sigmoid(hht)), 1)
