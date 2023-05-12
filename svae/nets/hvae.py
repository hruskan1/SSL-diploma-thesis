"""Hierarchical VAE"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict
import itertools

from ..custom_func.stoch_func import StochHeavi, ReSigm
#from custom_func.stoch_func import StochHeavi, ReSigm

from kornia.geometry.transform import Rescale,Resize
import warnings

from typing import Optional

def append_conv_block(mlist: nn.ModuleList, cfg: EasyDict, activation, **kwargs):
    chi = cfg.chi
    cho = cfg.cho
    kernel_size = cfg.k_size
    num_layer = getattr(cfg, 'num_l', 1)
    stride = getattr(cfg, 'stride', 1)
    pd = getattr(cfg, 'pad', 0)
    bnorm = getattr(cfg, 'bn', False)
    ll_act = getattr(cfg, 'll_act', False)
    # construct and append the layers
    for l in range(num_layer):
        cl = nn.Conv2d(chi, cho, kernel_size=kernel_size, stride=stride, padding=pd)
        mlist.append(cl)
        if bnorm:
            bnl = nn.BatchNorm2d(cho)
            mlist.append(bnl)
        mlist.append(activation)
        chi = cho

    # pop last activation [and bnorm]
    if not ll_act:
        mlist = mlist[:-1]
        # pop last bnorm 
        if bnorm: 
            mlist = mlist[:-1]

    return mlist

def append_conv_transpose_block(mlist: nn.ModuleList, cfg: EasyDict, activation, **kwargs):
    cfg = EasyDict(cfg)
    chi = cfg.chi
    cho = cfg.cho
    kernel_size = cfg.k_size
    num_layer = getattr(cfg, 'num_l', 1)
    stride = getattr(cfg, 'stride', 1)
    pd = getattr(cfg, 'pad', 0)
    bnorm = getattr(cfg, 'bn', False)
    ll_act = getattr(cfg, 'll_act', False)
    # construct and append the layers       
    for l in range(num_layer):
        if l < num_layer - 1:
            cl = nn.ConvTranspose2d(chi, chi, kernel_size=kernel_size, stride=stride, padding=pd)
            mlist.append(cl)
            if bnorm:
                bnl = nn.BatchNorm2d(chi)
                mlist.append(bnl)
        else:
            cl = nn.ConvTranspose2d(chi, cho, kernel_size=kernel_size, stride=stride, padding=pd)
            mlist.append(cl)
            if bnorm:
                bnl = nn.BatchNorm2d(cho)
                mlist.append(bnl)               
        mlist.append(activation)

    # pop last activation
    if not ll_act:
        mlist = mlist[:-1]
        # pop last bnorm
        if bnorm: 
            mlist = mlist[:-1]

    return mlist

def append_interpolation_block(mlist: nn.ModuleList, cfg: EasyDict, **kwargs):
    u = nn.Upsample(size=cfg.out_size)
    mlist.append(u)

    return mlist

def append_resize_block(mlist: nn.ModuleList, cfg: EasyDict, **kwargs):
    i = cfg.get('interpolation','bilinear')
    r = Resize(size=cfg.out_size,interpolation=i)
    mlist.append(r)

    return mlist

def append_rescale_block(mlist: nn.ModuleList, cfg: EasyDict, **kwargs):
    i = cfg.get('interpolation','bilinear')
    r = Rescale(factor=cfg.factor, interpolation=i)
    mlist.append(r)

    return mlist


def append_blocks(mlist: nn.ModuleList, bcfg: EasyDict, activation, **kwargs):
    for cfg in bcfg:
        if cfg.type == 'c':
            mlist = append_conv_block(mlist, cfg, activation, **kwargs)
        elif cfg.type == 't':
            mlist = append_conv_transpose_block(mlist, cfg, activation, **kwargs)
        elif cfg.type == 'i':
            mlist = append_interpolation_block(mlist, cfg, **kwargs)
        elif cfg.type == 'id':
            mlist.append(nn.Identity())
        elif cfg.type == 'myenc':
            mlist.append(MyEnc(cfg=cfg,**kwargs))
        elif cfg.type == 'mydec':
            mlist.append(MyDec(cfg=cfg,**kwargs))
        elif cfg.type == 'resize':
            mlist = append_resize_block(mlist,cfg,**kwargs)
        elif cfg.type == 'rescale':
            mlist = append_rescale_block(mlist,cfg,**kwargs)
        elif cfg.type == 'a2dpool':
            mlist.append(nn.AdaptiveAvgPool2d(cfg.spatial_size))
        else:
            raise Exception("Requested block type missing/not implemented")
    return mlist




def choose_activation(cfg: EasyDict):
    activation = None
    if cfg.activ == 'relu':
        activation = nn.ReLU()
    elif cfg.activ == 'softmax':
        activation = nn.Softmax2d()
    elif cfg.activ == 'sigmoid':
        activation = nn.Sigmoid()
    elif cfg.activ == 'resigm':
        activation = ReSigm()
    else:    
        raise NotImplementedError(f"Requested activation {cfg.active} not implemented")
    return activation

class MyDec(nn.Module):
    """ad-hoc decoder for enabling input of different shape"""
    def __init__(self,cfg,**kwargs):
        super(MyDec, self).__init__()
        
        self.net = nn.Sequential(
            *append_conv_block(nn.ModuleList(),cfg,activation=nn.ReLU())
        )

    def forward(self,input:list):
        #categorical is the first
        assert  len(input) == 2

        cat,bin = input[0],input[1]

        assert bin.ndim == 4 and bin.shape[2] == 1 and bin.shape[3] == 1
        assert cat.ndim == 4

        N,_,H,W = cat.shape
        bin_replicated = bin.repeat(1,1,H,W)

        z0 = torch.cat([cat,bin_replicated],dim=1)
        output = self.net(z0)

        #print(f"{output.shape=}")
        
        return output

class MyEnc(nn.Module):
    """ad-hoc decoder for enabling input of different shape"""
    def __init__(self,cfg,**kwargs):
        super(MyEnc, self).__init__()
        
        #categorical is the first
        self.cat = nn.Sequential(
            *append_blocks(nn.ModuleList(),cfg.cat,activation=choose_activation(cfg))
        )
        
        self.bin = nn.Sequential(
            *append_blocks(nn.ModuleList(),cfg.bin,activation=choose_activation(cfg)),
            
        )

    def forward(self,x):
        
        cat_output = self.cat(x)
        bin_output = self.bin(x)
    
        #categorical is the first
        return [cat_output , bin_output]


class GaussNPactivation(nn.Module):
    def __init__(self, cho, **kwargs):
        super(GaussNPactivation, self).__init__()
        self.vact = nn.Softplus()
    def forward(self, act):
        assert act.shape[1] % 2 == 0
        eta1, eta2 = torch.split(act,int(act.shape[1]/2), dim=1)
        eta2 = self.vact(eta2)
        eta = torch.cat((eta1, eta2,), dim=1)
        return eta


class StochBinBlock(nn.Module):
    def __init__(self, bcfg, **kwargs):
        super(StochBinBlock, self).__init__()
        self.cfg = EasyDict(bcfg)
        self.activation = choose_activation(self.cfg)
        self.cfg.d_skip = getattr(self.cfg, 'd_skip', False)
        self.cfg.e_skip = getattr(self.cfg, 'e_skip', False)
        # define the nets
        enc_list = nn.ModuleList()
        enc_list = append_blocks(enc_list, bcfg.enc, self.activation, **kwargs)
        dec_list = nn.ModuleList()
        dec_list = append_blocks(dec_list, bcfg.dec, self.activation, **kwargs)

        self.encoder = nn.Sequential(*enc_list)
        self.decoder = nn.Sequential(*dec_list)
        if self.cfg.d_skip:
            dec_skip_list = nn.ModuleList()
            dec_skip_list = append_blocks(dec_skip_list, bcfg.dec_skip, 
                                          self.activation, **kwargs)
            self.dec_skip_net = nn.Sequential(*dec_skip_list)
        if self.cfg.e_skip:
            enc_skip_list = nn.ModuleList()
            enc_skip_list = append_blocks(enc_skip_list, bcfg.enc_skip, 
                                          self.activation, **kwargs)
            self.enc_skip_net = nn.Sequential(*enc_skip_list)

        # define npactivation & bce
        self.npactivation = nn.Identity()
        self.bcel = nn.BCEWithLogitsLoss(reduction='none')
        # define buffers for statistics       
        self.register_buffer("jsdivs", torch.zeros(self.cfg.cho))
        self.register_buffer("jsdivs_av", torch.zeros(self.cfg.cho))


    def update_stats(self, acte, actd):
        with torch.no_grad():
            pe = torch.distributions.Bernoulli(logits=acte)
            pd = torch.distributions.Bernoulli(logits=actd)
            pm = torch.distributions.Bernoulli(0.5*(pe.probs + pd.probs))
            jsdivs = torch.distributions.kl_divergence(pe, pm) + torch.distributions.kl_divergence(pd, pm)
            jsdivs_av = jsdivs.mean([0, 2, 3])
            self.jsdivs_av.data.copy_(0.99 * self.jsdivs_av.data + 0.01 * jsdivs_av.detach().data)

            pe1 = torch.distributions.Bernoulli(probs=pe.probs.mean([0, 2, 3]))
            pd1 = torch.distributions.Bernoulli(probs=pd.probs.mean([0, 2, 3]))
            pm1 = torch.distributions.Bernoulli(0.5*(pe1.probs + pd1.probs))
            jsdivs = torch.distributions.kl_divergence(pe1, pm1) + torch.distributions.kl_divergence(pd1, pm1)
            self.jsdivs.data.copy_(0.99 * self.jsdivs.data + 0.01 * jsdivs.detach().data)
    
    def sample(self, z, acte=None, stats=False, verbose=False):
        self.eval()
        with torch.no_grad():
            actd = self.decoder.forward(z)
            if self.cfg.d_skip:
                actd = actd + self.dec_skip_net(z)
            if acte is not None:
                act = actd + acte
            else:
                act = actd
            pd = torch.distributions.Bernoulli(logits=act)
            z_out = pd.sample()
            z_out = z_out.detach().clone()
            if stats:
                self.update_stats(actd=actd.detach().clone(), 
                                acte=act.detach().clone())
        return (z_out, pd.probs.detach().clone()) if verbose else z_out
    
    # Not used
    def enc_forward(self, x):
        y = self.npactivation(x)
        act = self.encoder.forward(y)
        if self.cfg.e_skip:
            act = act + self.enc_skip_net(y)
        return act

    # Not used
    def dec_forward(self, z, acte=None):
        stoch_heavy = StochHeavi()
        actd = self.decoder.forward(z)
        if self.cfg.d_skip:
            actd = actd + self.dec_skip_net(z)
        if acte is not None:
            act = actd + acte
        else:
            act = actd
        return stoch_heavy(act)
    
    def neg_llik(self, z_in, z_out, acte=None):
        actd = self.decoder.forward(z_in)
        if self.cfg.d_skip:
            actd = actd + self.dec_skip_net(z_in)
        if acte is not None:
            act = actd + acte
        else:
            act = actd
        nll = self.bcel(act, z_out).mean(dim=0).sum()
        return nll

    def init_weights(self):
        for m in self.decoder.children():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if isinstance(self.activation, nn.ReLU):
                    nn.init.kaiming_normal_(m.weight)
                else:   # sigmoid
                    nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.zeros_(m.bias)
        for m in self.encoder.children():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if isinstance(self.activation, nn.ReLU):
                    nn.init.kaiming_normal_(m.weight)
                else:   # sigmoid
                    nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.zeros_(m.bias)


class StochCatBlock(nn.Module):
    def __init__(self, bcfg, **kwargs):
        super(StochCatBlock, self).__init__()
        self.cfg = EasyDict(bcfg)
        self.activation = choose_activation(self.cfg)
        self.cfg.d_skip = getattr(self.cfg, 'd_skip', False)
        self.cfg.e_skip = getattr(self.cfg, 'e_skip', False)
        # define the nets
        enc_list = nn.ModuleList()
        enc_list = append_blocks(enc_list, bcfg.enc, self.activation, **kwargs)
        dec_list = nn.ModuleList()
        dec_list = append_blocks(dec_list, bcfg.dec, self.activation, **kwargs)
        self.encoder = nn.Sequential(*enc_list)
        self.decoder = nn.Sequential(*dec_list)
        if self.cfg.d_skip:
            dec_skip_list = nn.ModuleList()
            dec_skip_list = append_blocks(dec_skip_list, bcfg.dec_skip, 
                                          self.activation, **kwargs)
            self.dec_skip_net = nn.Sequential(*dec_skip_list)
        if self.cfg.e_skip:
            enc_skip_list = nn.ModuleList()
            enc_skip_list = append_blocks(enc_skip_list, bcfg.enc_skip, 
                                          self.activation, **kwargs)
            self.enc_skip_net = nn.Sequential(*enc_skip_list)
        # define npactivation & loss
        self.npactivation = nn.Identity()
        self.cel = nn.CrossEntropyLoss(reduction='none')
        # define buffers for statistics       
        self.register_buffer("jsdivs", torch.zeros(1))
        self.register_buffer("jsdivs_av", torch.zeros(1))

    def update_stats(self, actd, acte):
        with torch.no_grad():
            pe = torch.distributions.OneHotCategorical(logits=acte.movedim(1, -1))
            pd = torch.distributions.OneHotCategorical(logits=actd.movedim(1, -1))
            pm = torch.distributions.OneHotCategorical(probs=0.5*(pe.probs + pd.probs))
            jsdivs = torch.distributions.kl_divergence(pe, pm) + torch.distributions.kl_divergence(pd, pm)
            jsdivs_av = jsdivs.mean()
            self.jsdivs_av.data.copy_(0.99 * self.jsdivs_av.data + 0.01 * jsdivs_av.detach().data)

            pe1 = torch.distributions.OneHotCategorical(probs=pe.probs.mean([0, 1, 2]))
            pd1 = torch.distributions.OneHotCategorical(probs=pd.probs.mean([0, 1, 2]))
            pm1 = torch.distributions.OneHotCategorical(probs=0.5*(pe1.probs + pd1.probs))
            jsdivs = torch.distributions.kl_divergence(pe1, pm1) + torch.distributions.kl_divergence(pd1, pm1)
            self.jsdivs.data.copy_(0.99 * self.jsdivs.data + 0.01 * jsdivs.detach().data)

    def sample(self, z, acte=None, stats=False, verbose=False):
        self.eval()
        with torch.no_grad():
            actd = self.decoder.forward(z)
            if self.cfg.d_skip:
                actd = actd + self.dec_skip_net(z)
            if acte is not None:
                act = actd + acte
            else:
                act = actd
            pd = torch.distributions.OneHotCategorical(logits=act.movedim(1, -1))
            z_out = pd.sample().movedim(-1, 1)
            z_out = z_out.detach().clone()
            if stats:
                self.update_stats(actd=actd.detach().clone(), 
                                acte=act.detach().clone())                
        return (z_out, pd.probs.movedim(-1, 1).detach().clone()) if verbose else z_out  

    def neg_llik(self, z_in, z_out, acte=None):
        actd = self.decoder.forward(z_in)
        if self.cfg.d_skip:
                actd = actd + self.dec_skip_net(z_in)
        if acte is not None:
            act = actd + acte
        else:
            act = actd
        nll = self.cel(act, z_out).mean(dim=0).sum()
        return nll

    def init_weights(self):
        for m in self.decoder.children():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if isinstance(self.activation, nn.ReLU):
                    nn.init.kaiming_normal_(m.weight)
                else:   # sigmoid
                    nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.zeros_(m.bias)
        for m in self.encoder.children():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if isinstance(self.activation, nn.ReLU):
                    nn.init.kaiming_normal_(m.weight)
                else:   # sigmoid
                    nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.zeros_(m.bias)   


class StochGaussBlock(nn.Module):
    def __init__(self, bcfg, **kwargs):
        super(StochGaussBlock, self).__init__()
        self.cfg = EasyDict(bcfg)
        self.activation = choose_activation(self.cfg)
        self.cfg.d_skip = getattr(self.cfg, 'd_skip', False)
        self.cfg.e_skip = getattr(self.cfg, 'e_skip', False)
        # define the nets
        enc_list = nn.ModuleList()
        enc_list = append_blocks(enc_list, bcfg.enc, self.activation, **kwargs)
        dec_list = nn.ModuleList()
        dec_list = append_blocks(dec_list, bcfg.dec, self.activation, **kwargs)

        self.encoder = nn.Sequential(*enc_list)
        self.decoder = nn.Sequential(*dec_list)
        if self.cfg.d_skip:
            dec_skip_list = nn.ModuleList()
            dec_skip_list = append_blocks(dec_skip_list, bcfg.dec_skip, 
                                          self.activation, **kwargs)
            self.dec_skip_net = nn.Sequential(*dec_skip_list)
        if self.cfg.e_skip:
            enc_skip_list = nn.ModuleList()
            enc_skip_list = append_blocks(enc_skip_list, bcfg.enc_skip, 
                                          self.activation, **kwargs)
            self.enc_skip_net = nn.Sequential(*enc_skip_list)
        # define npactivation & loss
        self.npactivation = GaussNPactivation(self.cfg.cho)
        
        self.gnnnl = nn.GaussianNLLLoss(full=True, reduction='none')
        # define buffers for statistics       
        self.register_buffer("jsdivs", torch.zeros(self.cfg.cho))
        self.register_buffer("jsdivs_av", torch.zeros(self.cfg.cho))

    def compute_st_param(self, act):
        # computes mu and var from natural parameters
        eta = self.npactivation(act)
        eta1, eta2 = torch.split(eta, self.cfg.cho, dim=1)
        mu = eta1.div(2.0 * eta2)
        var = torch.pow(2 * eta2, -1)
        return mu, var

    def update_stats(self, actd, acte):
        with torch.no_grad():
            mud, vard = self.compute_st_param(actd)
            sigd = vard.sqrt()
            mue, vare = self.compute_st_param(acte)
            sige = vare.sqrt()
            pd = torch.distributions.Normal(mud, sigd)
            pe = torch.distributions.Normal(mue, sige)
            # surrogate Gaussian (for mixture)
            mum = 0.5 * mud + 0.5 * mue
            varm = 0.5 * (vard + vare + 0.5 * (mud - mue).square())
            sigm = varm.sqrt()
            pm = torch.distributions.Normal(mum, sigm)
            jsdivs = torch.distributions.kl_divergence(pe, pm) + torch.distributions.kl_divergence(pd, pm)
            jsdivs_av = jsdivs.mean([0, 2, 3])
            self.jsdivs_av.data.copy_(0.99 * self.jsdivs_av.data + 0.01 * jsdivs_av.detach().data)

            sig_e_av, mu_e_av = torch.std_mean(pe.sample(), (0, 2, 3))
            sig_d_av, mu_d_av = torch.std_mean(pd.sample(), (0, 2, 3))
            sig_m_av, mu_m_av = torch.std_mean(pm.sample(), (0, 2, 3))
     
            pe1 = torch.distributions.Normal(mu_e_av, sig_e_av)
            pd1 = torch.distributions.Normal(mu_d_av, sig_d_av)
            pm1 = torch.distributions.Normal(mu_m_av, sig_m_av)
            jsdivs = torch.distributions.kl_divergence(pe1, pm1) + torch.distributions.kl_divergence(pd1, pm1)
            self.jsdivs.data.copy_(0.99 * self.jsdivs.data + 0.01 * jsdivs.detach().data)   

    def sample(self, z, acte=None, stats=False, verbose=False):
        self.eval()
        with torch.no_grad():
            actd = self.decoder.forward(z)
            if self.cfg.d_skip:
                actd = actd + self.dec_skip_net(z)
            if acte is not None:
                act = actd + acte
            else:
                act = actd
            mu, var = self.compute_st_param(act)
            sig = var.sqrt()
            pd = torch.distributions.Normal(mu, sig)
            z_out = pd.sample()
            z_out = z_out.detach().clone()
            if stats:
                self.update_stats(actd=actd.detach().clone(), 
                                acte=act.detach().clone())
        return (z_out, mu.detach().clone()) if verbose else z_out

    def neg_llik(self, z_in, z_out, acte=None):
        actd = self.decoder.forward(z_in)
        if self.cfg.d_skip:
            actd = actd + self.dec_skip_net(z_in)
        if acte is not None:
            act = actd + acte
        else:
            act = actd
        mu, var = self.compute_st_param(act)
        nll = self.gnnnl(mu, z_out, var).mean(dim=0).sum()
        return nll

    def init_weights(self):
        for m in self.decoder.children():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if isinstance(self.activation, nn.ReLU):
                    nn.init.kaiming_normal_(m.weight)
                else:   # sigmoid
                    nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.zeros_(m.bias)
        for m in self.encoder.children():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if isinstance(self.activation, nn.ReLU):
                    nn.init.kaiming_normal_(m.weight)
                else:   # sigmoid
                    nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.zeros_(m.bias)


class BinPriorBlock(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(BinPriorBlock, self).__init__()
        self.cfg = EasyDict(cfg)
        # define sigmoid, bce & npactivation
        self.sigmoid = nn.Sigmoid()
        self.bcel = nn.BCEWithLogitsLoss(reduction='none')
        self.npactivation = nn.Identity()
        # define buffers for statistics       
        self.register_buffer("jsdivs", torch.zeros(self.cfg.cho))
        self.register_buffer("jsdivs_av", torch.zeros(self.cfg.cho))

    def update_stats(self, acte):
        with torch.no_grad():
            pe = torch.distributions.Bernoulli(logits=acte)
            pd = torch.distributions.Bernoulli(logits=torch.zeros_like(acte))
            pm = torch.distributions.Bernoulli(probs=0.5*(pe.probs + pd.probs))
            jsdivs = torch.distributions.kl_divergence(pe, pm) + torch.distributions.kl_divergence(pd, pm)

            #jsdivs_av = jsdivs.mean([0, 2, 3])
            jsdivs_av = jsdivs.mean([0,*range(-jsdivs.ndim+2,0)])
            self.jsdivs_av.data.copy_(0.99 * self.jsdivs_av.data + 0.01 * jsdivs_av.detach().data)

            pe1 = torch.distributions.Bernoulli(probs=pe.probs.mean([0,*range(-jsdivs.ndim+2,0)]))
            pd1 = torch.distributions.Bernoulli(probs=pd.probs.mean([0,*range(-jsdivs.ndim+2,0)]))
            pm1 = torch.distributions.Bernoulli(probs=0.5*(pe1.probs + pd1.probs))
            jsdivs = torch.distributions.kl_divergence(pe1, pm1) + torch.distributions.kl_divergence(pd1, pm1)
            self.jsdivs.data.copy_(0.99 * self.jsdivs.data + 0.01 * jsdivs.detach().data)
    
    def sample_prior(self, shape):
        with torch.no_grad():  # this is perhaps not needed here
            z = torch.empty(shape).to(self.jsdivs.device).random_(2)
        return z.detach().clone()

    def sample(self, acte, stats=False, verbose=False):
        with torch.no_grad():
            pd = torch.distributions.Bernoulli(logits=acte)
            z = pd.sample().detach().clone()
            if stats:
                self.update_stats(acte.detach().clone())
        return (z, pd.probs.detach().clone()) if verbose else z

    def neg_llik(self, z_out, acte):
        nll = self.bcel(acte, z_out).mean(dim=0).sum()
        return nll


class CatPriorBlock(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(CatPriorBlock, self).__init__()
        self.cfg = cfg
        # define cel & npactivation
        
        if 'class_weights' in kwargs:
            self.cel = nn.CrossEntropyLoss(reduction='none',weight=kwargs['class_weights'])
        else:
            self.cel = nn.CrossEntropyLoss(reduction='none')

        self.npactivation = nn.Identity()
        # define buffers for statistics       
        self.register_buffer("jsdivs", torch.zeros(1))
        self.register_buffer("jsdivs_av", torch.zeros(1))

    def update_stats(self, acte):
        with torch.no_grad():
            pe = torch.distributions.OneHotCategorical(logits=acte.movedim(1, -1))
            pd = torch.distributions.OneHotCategorical(logits=torch.zeros_like(acte.movedim(1, -1)))
            pm = torch.distributions.OneHotCategorical(probs=0.5*(pe.probs + pd.probs))
            jsdivs = torch.distributions.kl_divergence(pe, pm) + torch.distributions.kl_divergence(pd, pm)
            jsdivs_av = jsdivs.mean()
            self.jsdivs_av.data.copy_(0.99 * self.jsdivs_av.data + 0.01 * jsdivs_av.detach().data)

            pe1 = torch.distributions.OneHotCategorical(probs=pe.probs.mean([0, 1, 2]))
            pd1 = torch.distributions.OneHotCategorical(probs=pd.probs.mean([0, 1, 2]))
            pm1 = torch.distributions.OneHotCategorical(probs=0.5*(pe1.probs + pd1.probs))
            jsdivs = torch.distributions.kl_divergence(pe1, pm1) + torch.distributions.kl_divergence(pd1, pm1)
            self.jsdivs.data.copy_(0.99 * self.jsdivs.data + 0.01 * jsdivs.detach().data)
    
    def sample_prior(self, shape):
        with torch.no_grad():  # this is perhaps not necessary here
            pd = torch.distributions.OneHotCategorical(logits=torch.zeros(shape).to(self.jsdivs.device).movedim(1,-1))
            z = pd.sample().movedim(-1, 1)
        return z.detach().clone()

    def sample(self, acte, stats=False, verbose=False):
        with torch.no_grad():
            pd = torch.distributions.OneHotCategorical(logits=acte.movedim(1, -1))
            z = pd.sample().movedim(-1, 1)
            z = z.detach().clone()
            if stats:
                self.update_stats(acte.detach().clone())   
        return (z, pd.probs.movedim(-1, 1).detach().clone()) if verbose else z

    def neg_llik(self, z_out, acte):
        nll = self.cel(acte, z_out).mean(dim=0).sum()
        return nll


class GaussPriorBlock(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(GaussPriorBlock, self).__init__()
        self.cfg = EasyDict(cfg)
        # define buffers for statistics       
        self.register_buffer("jsdivs", torch.zeros(self.cfg.cho))
        self.register_buffer("jsdivs_av", torch.zeros(self.cfg.cho))
        self.npactivation = GaussNPactivation(self.cfg.cho)
        self.gnnnl = nn.GaussianNLLLoss(full=True, reduction='none')

    def compute_st_param(self, act):
        # computes mu and var from natural parameters
        eta = self.npactivation(act)
        eta1, eta2 = torch.split(eta, self.cfg.cho, dim=1)
        mu = eta1.div(2.0 * eta2)
        var = torch.pow(2 * eta2, -1)
        return mu, var    

    def update_stats(self, act):
        with torch.no_grad():
            mu, var = self.compute_st_param(act)
            sig = var.sqrt()
            pe = torch.distributions.Normal(mu, sig)
            pd = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sig))
            # surrogate Gaussian (for mixture)
            mum = 0.5 * mu
            varm = 0.5 * sig.square().add(0.5) + 0.25 * mu.square()
            sigm = varm.sqrt()
            pm = torch.distributions.Normal(mum, sigm)
            jsdivs = torch.distributions.kl_divergence(pe, pm) + torch.distributions.kl_divergence(pd, pm)
            jsdivs_av = jsdivs.mean([0, 2, 3])
            self.jsdivs_av.data.copy_(0.99 * self.jsdivs_av.data + 0.01 * jsdivs_av.detach().data)

            sig_e_av, mu_e_av  = torch.std_mean(pe.sample(), (0, 2, 3))
            sig_d_av, mu_d_av  = torch.std_mean(pd.sample(), (0, 2, 3))
            sig_m_av, mu_m_av  = torch.std_mean(pm.sample(), (0, 2, 3))
     
            pe1 = torch.distributions.Normal(mu_e_av, sig_e_av)
            pd1 = torch.distributions.Normal(mu_d_av, sig_d_av)
            pm1 = torch.distributions.Normal(mu_m_av, sig_m_av)
            jsdivs = torch.distributions.kl_divergence(pe1, pm1) + torch.distributions.kl_divergence(pd1, pm1)
            self.jsdivs.data.copy_(0.99 * self.jsdivs.data + 0.01 * jsdivs.detach().data)            


    def sample_prior(self, shape):
        self.eval()
        with torch.no_grad():
            z = torch.randn(shape).to(self.jsdivs.device)
        return z.detach().clone()
    
    def sample(self, acte, stats=False, verbose=False):
        with torch.no_grad():
            mu, var = self.compute_st_param(acte)
            sig = var.sqrt()
            pd = torch.distributions.Normal(mu, sig)
            z = pd.sample()
            z = z.detach().clone()
            if stats:
                self.update_stats(acte.detach().clone())
        return (z, mu.detach().clone()) if verbose else z
   
    def neg_llik(self, z_out, acte):
        mu, var = self.compute_st_param(acte)
        nll = self.gnnnl(mu, z_out, var).mean(dim=0).sum()
        return nll

class CustomPriorBlock(nn.Module):
    """Custom prior block for combination of segmentation (categorical) and latent (binary) variables
    
        categorical is the first
        bernulli is the second
    """
    def __init__(self,cfg):

        super(CustomPriorBlock, self).__init__()
        self.cfg = EasyDict(cfg)
        self.cat = CatPriorBlock(cfg.pc_block,class_weights=cfg.class_weights)
        self.bin = BinPriorBlock(cfg.bin_block)

        self.jsdivs = (self.cat.jsdivs, self.bin.jsdivs)
        self.jsdivs_av = (self.cat.jsdivs_av,self.bin.jsdivs_av)


    def update_stats(self, acte):
        cat_acte,bin_acte = acte[0],acte[1]
        self.cat.update_stats(cat_acte)
        self.bin.update_stats(bin_acte)

    
    def sample_prior(self, shapes):
        cat_shape,bin_shape = shapes[0],shapes[1]
        z0_cat = self.cat.sample_prior(cat_shape)
        z0_bin = self.bin.sample_prior(bin_shape)
        return [z0_cat, z0_bin]

    def sample(self, acte, stats=False, verbose=False):
        cat = self.cat.sample(acte[0], stats, verbose)
        bin = self.bin.sample(acte[1], stats, verbose)
        if verbose:
            z0_cat,cat_probs = cat
            z0_bin,bin_probs = bin

            return [z0_cat,z0_bin],[cat_probs,bin_probs]
        else:
            return [cat,bin]
        
        
    def neg_llik(self, z_out, acte):
        nll = self.cat.neg_llik(z_out[0],acte=acte[0])
        nll += self.bin.neg_llik(z_out[1],acte=acte[1])

        return nll
    


class HVAE(nn.Module):
    """High-level module for Hiarchical VAEs"""
    def __init__(self, **kwargs):
        super(HVAE, self).__init__()

        # Get DAG matrix representing the skip connections 
        self.graph_matrix = np.array(kwargs['graph_matrix'])
        assert self.graph_matrix.shape[0] == len(kwargs['blocks'])

        self.compute_block_channels(**kwargs) # alternates config file 
         
        bcfg = kwargs['blocks']
        lparm = kwargs['lparm']

        self.z0_shape = None
        self.sblock_list = nn.ModuleList()

        # prior block
        cfg = bcfg[0]
        if cfg['type'] == 'pc':
            bl = CatPriorBlock(cfg)
        elif cfg['type'] == 'pb':
            bl = BinPriorBlock(cfg)
        elif cfg['type'] == 'pg':
            bl = GaussPriorBlock(cfg)
        elif cfg['type'] == 'pcustom':
            bl = CustomPriorBlock(cfg)
        else:
            raise Exception("Requested prior block type missing/not implemented")
        self.sblock_list.append(bl)
        
        # all other latent blocks & image block
        dec_paraml = []
        enc_paraml = []
        for cfg in bcfg[1:]:
            if cfg['type'] == 'sb':
                bl = StochBinBlock(cfg, **kwargs)
            elif cfg['type'] == 'sc':
                bl = StochCatBlock(cfg, **kwargs)
            elif cfg['type'] == 'sg':
                bl = StochGaussBlock(cfg, **kwargs)
            else:
               Exception("Requested stochastic block type missing/not implemented") 
            self.sblock_list.append(bl)
            dec_paraml.append(bl.decoder.parameters())
            enc_paraml.append(bl.encoder.parameters())
            if cfg.d_skip:
                dec_paraml.append(bl.dec_skip_net.parameters())
            if cfg.e_skip:
                enc_paraml.append(bl.enc_skip_net.parameters())
        dec_parameters = itertools.chain(*dec_paraml)
        enc_parameters = itertools.chain(*enc_paraml)
        self.dec_optimizer = torch.optim.Adam(dec_parameters, lr=lparm['dec_stepsize'])
        self.enc_optimizer = torch.optim.Adam(enc_parameters, lr=lparm['enc_stepsize'])


    def encoder_eval(self):
        for sblock in self.sblock_list[1:]:
            sblock.encoder.eval()
    
    def encoder_train(self):
        for sblock in self.sblock_list[1:]:
            sblock.encoder.train()
    
    def decoder_eval(self):
        for sblock in self.sblock_list[1:]:
            sblock.decoder.eval()

    def decoder_train(self):
        for sblock in self.sblock_list[1:]:
            sblock.decoder.train()

    def encoder_requires_grad(self, on=True):
        for sblock in self.sblock_list[1:]:
           sblock.encoder.requires_grad_(on)
    
    def decoder_requires_grad(self, on=True):
        for sblock in self.sblock_list[1:]:
           sblock.decoder.requires_grad_(on)    

    def z0_prior_sample(self, shape):
        sblock = self.sblock_list[0]
        z = sblock.sample_prior(shape)
        return z

    def prior_sample(self, z0, expanded=False):
        """Function which samples x (all latent zs) the prior for given z0 """
        self.eval()
        num_blocks = len(self.sblock_list)
        z_all = [None] * num_blocks
        z = z0
        with torch.no_grad():
            z_all[0] = z
            # sample from other latent blocks
            for idx,sblock in zip(range(1,num_blocks-1),self.sblock_list[1:-1]):
                z_input = self._get_layer_inputs(idx,z_all,is_encoder=False)
                z = sblock.sample(z_input)
                z_all[idx] = z

            # cprobs & sample image
            idx,sblock = num_blocks-1,self.sblock_list[-1]
            z_input = self._get_layer_inputs(idx,z_all,is_encoder=False)
            z, cprobs = sblock.sample(z_input, verbose=True)
            z_all[-1] = z

        return (z_all, cprobs) if expanded else (z, cprobs)

    def encoder_activations(self, x):
        """Function which activates the encoder with input x (going in reverse direction)"""

        # Intilize enc_activations to None
        num_blocks = len(self.sblock_list)
        enc_acts = [None] * num_blocks

        # Propagate x through last block (first in encoder direction), 
        y = x
        enc_acts[-1] = None
        
        sblock = self.sblock_list[-1]

        act = sblock.encoder.forward(y)
        if sblock.cfg.e_skip:
            act = act + sblock.enc_skip_net(y)

        enc_acts[-2] = act

        # print(f"-1. output {enc_acts[-1].shape=}")
        # print(f"-2. output {enc_acts[-2].shape=}")

        # Propagate inputs through the encoder in reverse direction (omit last block as it is only prior)
        for idx,sblock in zip(reversed(range(1,num_blocks-1)),reversed(self.sblock_list[1:-1])): 
        
            # Concat all npactivations  
            layer_inputs = self._get_layer_inputs(idx,enc_acts,is_encoder=True)
            
            # should be independent on npactivation
            y = sblock.npactivation(layer_inputs) 
            act = sblock.encoder.forward(y)
            if sblock.cfg.e_skip:
                act = act + sblock.enc_skip_net(y)
            
            # if type(act) in (list, tuple):
            #     act_shape = act[0].shape, act[1].shape
            # else:
            #     act_shape = act.shape
            # print(f"{idx-1}. {layer_inputs.shape=}")
            # print(f"{idx-1}. output_{act_shape=}")
            
            enc_acts[idx-1] = act
            
        
        # Return encoder activations
        return enc_acts

    def posterior_sample(self, x0_smpl, z0=None, stats=False, expanded=False):
        # notice that this function samples the latent variables and a new image
        self.eval()
        num_blocks = len(self.sblock_list)
        z_all = [None] * num_blocks
        probs = []
        with torch.no_grad():
            # apply encoder
            enc_acts = self.encoder_activations(x0_smpl)
            # sample latents
            # prior layer
            sblock = self.sblock_list[0]
            if z0 is None:
                z, p = sblock.sample(enc_acts[0], stats=stats, verbose=True)
                
                cprobs = p.detach().clone() if not (type(p) in (list, tuple)) else [p[0].detach().clone(),p[1].detach().clone()]
                probs.append(p)
            else:
                # if z0 from multiple blocks (in list/tuple)
                z = z0.detach().clone() if not (type(z0) in (list, tuple)) else [z0[0].detach().clone(),z0[1].detach().clone()]
            z_all[0] = z
            

            # other latent layers
            # sample from other latent blocks

            for idx,sblock, act in zip(range(1,num_blocks-1),self.sblock_list[1:-1], enc_acts[1:-1]):
                z_input = self._get_layer_inputs(idx,z_all,is_encoder=False)
                z = sblock.sample(z_input, acte=act, stats=stats)
                z_all[idx] = z

            # cprobs & sample image
            idx, sblock = num_blocks-1,self.sblock_list[-1]
            z_input = self._get_layer_inputs(idx,z_all,is_encoder=False)
            z, cprobs = sblock.sample(z_input, verbose=True)
            z_all[-1] = z
            probs.append(cprobs.detach().clone())
        z0 = z_all[0]
        self.z0_shape = z.shape if not (type(z0) in (list, tuple)) else [z0[0].shape,z0[1].shape]

        return (z_all, probs) if expanded else (z, probs)

    def decoder_learn_step(self, x0_smpl, z0=None):
        # get posterior sample
        (z_all, _) = self.posterior_sample(x0_smpl, z0=z0, stats=True, expanded=True)
        # replace last z by images:
        z_all[-1] = x0_smpl.detach()
        # set attributes
        self.decoder_train()
        self.encoder_eval()
        self.dec_optimizer.zero_grad()
        # accumulate loss
        loss = torch.zeros(1).to(next(self.sblock_list[-1].decoder.parameters()).device)
        # latent blocks + image block
        for idx,sblock, z_out in zip(range(1,len(self.sblock_list)),self.sblock_list[1:], z_all[1:]):
            z_in = self._get_layer_inputs(idx,z_all,is_encoder=False)
            nll = sblock.neg_llik(z_in=z_in, z_out=z_out)
            loss = loss + nll

        loss.backward()
        self.dec_optimizer.step()
        return loss.detach()

    def encoder_learn_step(self, z0):
        # get prior sample
        (z_all, _) = self.prior_sample(z0, expanded=True)
        # set attributes
        self.decoder_eval()
        self.encoder_train()
        self.enc_optimizer.zero_grad()
        # apply encoder
        enc_act = self.encoder_activations(z_all[-1])
        # learn steps
        # prior block
        sblock = self.sblock_list[0]
        loss = sblock.neg_llik(z_all[0], enc_act[0])

        # other latent blocks
        for idx,sblock, z_out, acte in zip(range(1,len(self.sblock_list)-1), self.sblock_list[1:-1], z_all[1:], enc_act[1:-1]):
            z_in = self._get_layer_inputs(idx,z_all,is_encoder=False)
            #print(f"{idx=},{z_in.shape=}")
            nll = sblock.neg_llik(z_in, z_out, acte=acte)
            loss = loss + nll
        loss.backward()
        self.enc_optimizer.step()
        return loss.detach()
    
    def encoder_supervised_learn_step(self, x0, z0):
        # set attributes
        self.encoder_train()
        self.enc_optimizer.zero_grad()
        # apply encoder
        enc_act = self.encoder_activations(x0)
        # learn steps
        # prior block
        sblock = self.sblock_list[0]
        loss = sblock.neg_llik(z0, enc_act[0])
        loss.backward()
        self.enc_optimizer.step()
        return loss.detach()

    def save_stats(self, path):
        vdict = {}
        for i, sblock in enumerate(self.sblock_list[:-1]):
            if type(sblock.jsdivs_av) in (list, tuple):
                for j,avg in enumerate(sblock.jsdivs_av):
                    vdict[f"{i}.{j}"] = sblock.jsdivs_av[j].detach().cpu().numpy()
            else:
                vdict[f"{i}"] = sblock.jsdivs_av.detach().cpu().numpy()
        np.savez_compressed(path, **vdict)       

    def init_weights(self):
        for sblock in self.sblock_list[1:]:
            sblock.init_weights()


    def compute_block_channels(self,**kwargs):
        """
        Compute the correct number of channels for the given encoder and decoder block so the encoder_activation function and sampling can be run.
        The connections between blocks are encoded by the `graph_matrix` and we assume that they are symetric with respect to encoder and decoder.
        So if Enc[i] has connection to Enc[j], then Dec[j] is connected to Dec[i].
        """

        bcfg = kwargs['blocks']

        # Iterate over blocks (but not prior)
        for idx,cfg in enumerate(bcfg[1:]):
            i = idx + 1
            
            # Get encoder & decoder correct size
            enc_channels,dec_channels =  0,0
            for j in range(len(bcfg)):

                if self.graph_matrix[j,i] == 1:
                    enc_channels += bcfg[j].get('chi',float('nan'))

                if self.graph_matrix[i,j] == 1:
                    dec_channels += bcfg[j].get('cho',float('nan'))

            enc_chi = _get_first_convolution(cfg.enc).chi
            dec_chi = _get_first_convolution(cfg.dec).chi
           

            if enc_chi != enc_channels and i < (len(bcfg) - 1): # last block's encoder accepts x. 
                warnings.warn(f'{i}. block: encoder first layer input channels do not correspond to graph matrix: {enc_chi} instead of {enc_channels}')
                _get_first_convolution(cfg.enc).chi = enc_channels
 

            if dec_chi != dec_channels:
                warnings.warn(f'{i}. block: decoder first layer input channels do not correspond to graph matrix: {dec_chi} instead of {dec_channels}')
                _get_first_convolution(cfg.dec).chi = dec_channels
            
            if cfg.e_skip:
                enc_skip_conv = _get_first_convolution(cfg.enc_skip)
                if enc_skip_conv is not None and enc_skip_conv.chi != enc_channels  and i < (len(bcfg) - 1): # last block's encoder accepts x.
                    warnings.warn(f'{i}. block: encoder skip network first layer input channels do not correspond to graph matrix: {enc_skip_conv.chi} instead of {enc_channels}')
                    enc_skip_conv.chi = enc_channels
            
            if cfg.d_skip:
                dec_skip_conv = _get_first_convolution(cfg.dec_skip)
                if dec_skip_conv is not None and dec_skip_conv.chi != dec_channels:
                    warnings.warn(f'{i}. block: decoder skip network first layer input channels do not correspond to graph matrix: {dec_skip_conv.chi} instead of {dec_channels}')
                    dec_skip_conv.chi = dec_channels
                

            # To be removed
            # chi = cfg.get('chi',float('nan'))
            # cho = cfg.get('cho',float('nan')) 
            # in_size = cfg.get('in_size',float('nan')) 
            # out_size = cfg.get('out_size',float('nan'))    

            # print(f"{i}.|{chi=},{cho=},{in_size=},{out_size=},{enc_channels=},{dec_channels=}")
            # print(f"{cfg.dec=}")
            # print(f"{cfg.enc=}")
    

    def _get_layer_inputs(self,idx:int,activations:list[Optional[torch.Tensor]],is_encoder:bool=False)-> list[torch.Tensor|list]:
        """
        Concatenates inputs from different layers (specified by graph matrix)

        idx (int): number of the layer 
        activations (list): list of activations from the layers [x0,x1,x2,x3,x4...,xn] xi's shape is [N,C,H,W]
                             or list of  latent variables [z0,...,zi,..,zn], zi's shape is [N,C,H,W]
        is_encoder: boolean if True, compose layer input for the encoder, if False, compose layer input for the decoder
        """
        is_decoder = not(is_encoder)

        # TODO: vectorize 
        all_acts = []

        if is_encoder:
            for j in range(self.graph_matrix.shape[0]-1,-1,-1): # the first input is from the closest (previous) layer
                if self.graph_matrix[j,idx] == 1:
                    #print(j,idx)
                    all_acts.append(activations[j-1]) # -1 as it z_{i-1} is input to the i-th layer
        
        elif is_decoder:
            for j in range(self.graph_matrix.shape[1]-1,-1,-1): 
                if self.graph_matrix[idx,j] == 1:
                    #print(idx,j)
                    all_acts.append(activations[j]) # z_{i} is the output of the i-th layer

        if idx == 1 and is_decoder: # hot fix to pass special case from prior to first block, only one input, pop list
            all_acts = all_acts[0]
        else:
            # for a in all_acts:
            #     print(f"{a.shape=}")
            all_acts = torch.cat(all_acts,dim=-3)
            #print(f"{all_acts.shape=}")
        return all_acts

            
def _get_first_convolution(block):
    """Returns first convolution (or transposed convolution) for a block. If not found, returns None"""
    for l in block:
        if l.type == 'c' or l.type == 't' or l.type == 'mydec':
            return l 
        elif l.type == 'myenc':
            return _get_first_convolution(l.cat)
    return None