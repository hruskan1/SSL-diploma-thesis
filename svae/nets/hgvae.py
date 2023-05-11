"""Hierarchical Gaussian VAE"""

from tabnanny import verbose
import torch
import torch.nn as nn
import numpy as np

SIGMOID = True       # activation inside the blocks sigmoid/ReLU
MU_MIN = -1.0
MU_MAX = 1.0e+1
SIG_MIN = 1.0e-2
SIG_MAX = 1.0e+1

def shift_gparms(mu1, sig1, mu2, sig2):
    # add natural parameters for Gaussians and return mu, sigma
    sig1s = sig1.square()
    sig2s = sig2.square()
    sigs = sig1s * sig2s / (sig1s + sig2s)
    sig = sigs.sqrt()
    mu = (mu1 * sig2s + mu2 * sig1s) / (sig1s + sig2s)

    return mu, sig


class ConvBlock(nn.Sequential):
    def __init__(self, layers, **kwargs):
        if kwargs['activation'] == 'relu':
            activation = nn.ReLU() 
        else:
            activation = nn.Sigmoid()
        layer_list = []
        # construct the layers
        for prev_layer, layer in zip(layers, layers[1:]):
            chi = prev_layer['channels']
            cho = layer['channels']
            kernel_size = layer['kernel_size']
            stride = layer['stride']
            if layer['padding'] is not None:
                pd = layer['padding']
                layer_list.append(nn.ReplicationPad2d(pd))
            if stride == 1:
                cl = nn.Conv2d(chi, cho, kernel_size=kernel_size, stride=stride)
                layer_list.append(cl)
            else:
                av = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
                layer_list.append(av)
                cl = nn.Conv2d(chi, cho, kernel_size=1, stride=1)
                layer_list.append(cl)
            if kwargs['bnorm']:
                bnl = nn.BatchNorm2d(cho)
                layer_list.append(bnl)
            layer_list.append(activation)
        
        # pop last activation
        layer_list.pop()
        # pop last bnorm if required
        if kwargs['bnorm'] and  (not kwargs['bnorm_ll']): 
            layer_list.pop()
        super(ConvBlock, self).__init__(*layer_list)

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                if SIGMOID:
                    nn.init.xavier_normal_(m.weight, gain=3.6)
                else:
                    nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)


class ConvTransposeBlock(nn.Sequential):
    def __init__(self, layers, **kwargs):
        layers.reverse()
        # construct the list of layers
        if kwargs['activation'] == 'relu':
            activation = nn.ReLU() 
        else:
            activation = nn.Sigmoid()
        layer_list = []
        for next_layer, layer in zip(layers[1:], layers):
            cho = next_layer['channels']
            chi = layer['channels']
            kernel_size = layer['kernel_size']
            stride = layer['stride']
            if layer['padding'] is not None:
                pd = layer['padding']
                layer_list.append(nn.ReplicationPad2d(pd))
            if stride == 1:
                cl = nn.ConvTranspose2d(chi, cho, kernel_size=kernel_size, stride=stride)
                layer_list.append(cl)
            else:
                cl = nn.ConvTranspose2d(chi, cho, kernel_size=1, stride=1)
                layer_list.append(cl)
                ups = nn.Upsample(scale_factor=stride)
                layer_list.append(ups)
            if kwargs['bnorm']:
                bnl = nn.BatchNorm2d(cho)
                layer_list.append(bnl)
            layer_list.append(activation)

        # pop last activation
        layer_list.pop()
        # pop last bnorm if required
        if kwargs['bnorm'] and  (not kwargs['bnorm_ll']): 
            layer_list.pop()
        super(ConvTransposeBlock, self).__init__(*layer_list)
        layers.reverse()
    
    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.ConvTranspose2d):
                if SIGMOID:
                    nn.init.xavier_normal_(m.weight, gain=3.6)
                else:
                    nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)


class StochConvBlock(nn.Module):
    def __init__(self, layers, **kwargs):
        super(StochConvBlock, self).__init__()
        # double channels in last layer for mu, sigma
        self.cho = layers[-1]["channels"] 
        layers[-1]["channels"] = self.cho * 2
        # define the net
        self.block = ConvBlock(layers, **kwargs)


    def forward(self, x):
        scores = self.block(x)
        mu, sig = torch.split(scores, self.cho, dim=1)
        sig = torch.exp(sig)
        mu = torch.clamp(mu, min=MU_MIN, max=MU_MAX)
        sig = torch.clamp(sig, min=SIG_MIN, max=SIG_MAX)
        return mu, sig

    def init_weights(self):
        self.block.init_weights()


class StochPriorBlock(nn.Module):
    def __init__(self, layers, **kwargs):
        super(StochPriorBlock, self).__init__()
        self.cho = layers[-1]["channels"]
        # define buffers for statistics       
        self.register_buffer("kldivs", torch.zeros(self.cho))
        self.register_buffer("kldivs_av", torch.zeros(self.cho))        

    def update_stats(self, mu_e, sig_e, mu_d, sig_d):
        with torch.no_grad():
            pe = torch.distributions.Normal(mu_e, sig_e)
            pd = torch.distributions.Normal(mu_d, sig_d)
            kldivs_av = torch.distributions.kl_divergence(pe, pd).mean([0, 2, 3])
            e_smpl = pe.sample()
            d_smpl = pd.sample()
            mu_e_av = e_smpl.mean([0, 2, 3])
            sig_e_av = e_smpl.std([0, 2, 3])
            mu_d_av = d_smpl.mean([0, 2, 3])
            sig_d_av = d_smpl.std([0, 2, 3])
            pe1 = torch.distributions.Normal(mu_e_av, sig_e_av)
            pd1 = torch.distributions.Normal(mu_d_av, sig_d_av)
            kldivs = torch.distributions.kl_divergence(pe1, pd1)

            self.kldivs_av.data.copy_(0.99 * self.kldivs_av.data + 0.01 * kldivs_av.detach().data)
            self.kldivs.data.copy_(0.99 * self.kldivs.data + 0.01 * kldivs.detach().data)

    def sample(self, z_shape):
        self.eval()
        with torch.no_grad():
            z = torch.randn(z_shape).to(self.kldivs.device)

        return z.detach().clone()

    def sample_shifted(self, mu2, sig2, stats=False):
        self.eval()
        with torch.no_grad():
            mud = torch.zeros_like(mu2)
            sigd = torch.ones_like(sig2)
            mue, sige = shift_gparms(mud, sigd, mu2, sig2)
            pd = torch.distributions.Normal(mue, sige)
            z_out = pd.sample()

            if stats:
                self.update_stats(mue.detach().clone(), 
                                    sige.detach().clone(), 
                                    mud.detach().clone(), 
                                    sigd.detach().clone())
        
        return z_out.detach().clone()

    def neg_llik_shifted(self, z_out, mu2, sig2):
        pe = torch.distributions.Normal(loc=mu2, scale=sig2)
        nll = pe.log_prob(z_out).mean(dim=0).sum()

        return -nll     


class StochConvTransposeBlock(nn.Module):
    def __init__(self, layers, **kwargs):
        super(StochConvTransposeBlock, self).__init__()
        # double channels in last layer for mu, sigma
        self.cho = layers[0]["channels"] 
        layers[0]["channels"] = self.cho * 2
        # define the net
        self.block = ConvTransposeBlock(layers, **kwargs)
        # define buffers for statistics       
        self.register_buffer("kldivs", torch.zeros(self.cho))
        self.register_buffer("kldivs_av", torch.zeros(self.cho))

    def update_stats(self, mu_e, sig_e, mu_d, sig_d):
        with torch.no_grad():
            pe = torch.distributions.Normal(mu_e, sig_e)
            pd = torch.distributions.Normal(mu_d, sig_d)
            kldivs_av = torch.distributions.kl_divergence(pe, pd).mean([0, 2, 3])
            e_smpl = pe.sample()
            d_smpl = pd.sample()
            mu_e_av = e_smpl.mean([0, 2, 3])
            sig_e_av = e_smpl.std([0, 2, 3])
            mu_d_av = d_smpl.mean([0, 2, 3])
            sig_d_av = d_smpl.std([0, 2, 3])
            pe1 = torch.distributions.Normal(mu_e_av, sig_e_av)
            pd1 = torch.distributions.Normal(mu_d_av, sig_d_av)
            kldivs = torch.distributions.kl_divergence(pe1, pd1)

            self.kldivs_av.data.copy_(0.99 * self.kldivs_av.data + 0.01 * kldivs_av.detach().data)
            self.kldivs.data.copy_(0.99 * self.kldivs.data + 0.01 * kldivs.detach().data)

    def forward(self, z):
        scores = self.block(z)
        mu, sig = torch.split(scores, self.cho, dim=1)
        sig = torch.exp(sig)
        mu = torch.clamp(mu, min=MU_MIN, max=MU_MAX)
        sig = torch.clamp(sig, min=SIG_MIN, max=SIG_MAX)
        return mu, sig        

    def cprobs(self, z):
        mu, sigma = self.forward(z)
        pd = torch.distributions.Normal(mu, sigma)
        return pd

    def cprobs_shifted(self, z, mu2, sig2):
        mu1, sig1 = self.forward(z)
        mu, sig = shift_gparms(mu1, sig1, mu2, sig2)
        pd = torch.distributions.Normal(mu, sig)
        return pd

    def sample(self, z, verbose=False):
        self.eval()
        with torch.no_grad():
            mu, sigma = self.forward(z)
            pd = torch.distributions.Normal(mu, sigma)
            z_out = pd.sample().detach().clone()

        return (z_out, mu.detach().clone()) if verbose else z_out

    def sample_shifted(self, z, mu2, sig2, stats=False, verbose=False):
        self.eval()
        with torch.no_grad():
            mud, sigd = self.forward(z)
            mue, sige = shift_gparms(mud, sigd, mu2, sig2)
            pe = torch.distributions.Normal(mue, sige)
            z_out = pe.sample().detach().clone()
            if stats:
                self.update_stats(mue.detach().clone(), 
                                    sige.detach().clone(), 
                                    mud.detach().clone(), 
                                    sigd.detach().clone())
        
        return (z_out, mue.detach().clone()) if verbose else z_out       

    def neg_llik(self, z_in, z_out):
        mu, sigma = self.forward(z_in)
        pd = torch.distributions.Normal(mu, sigma)
        ll = pd.log_prob(z_out).mean(dim=0).sum()

        return -ll     

    def neg_llik_shifted(self, z_in, z_out, mu2, sig2):
        mud, sigd = self.forward(z_in)
        mue, sige = shift_gparms(mud, sigd, mu2, sig2)
        pe = torch.distributions.Normal(mue, sige)
        ll = pe.log_prob(z_out).mean(dim=0).sum()

        return -ll

    def init_weights(self):
        self.block.init_weights()


class HGVAE(nn.Module):
    def __init__(self, **kwargs):
        super(HGVAE, self).__init__()
        enc_blocks = kwargs["enc_blocks"]
        enc_parm = kwargs["enc_parm"]
        dec_blocks = kwargs['dec_blocks']
        dec_parm = kwargs['dec_parm']
        # encoder
        self.eblock_list = nn.ModuleList()
        for eblock in enc_blocks:
            bl = StochConvBlock(eblock, **enc_parm)
            self.eblock_list.append(bl)
        self.enc_optimizer = torch.optim.Adam(self.eblock_list.parameters(), lr=enc_parm['stepsize'])
        # decoder
        self.dblock_list = nn.ModuleList()
        dec_blocks.reverse()

        # prior block
        dblock = dec_blocks[0]
        bl = StochPriorBlock(dblock, **dec_parm)
        self.dblock_list.append(bl)
        # all other latent blocks & image block
        for dblock in dec_blocks:
            bl = StochConvTransposeBlock(dblock, **dec_parm)
            self.dblock_list.append(bl)
        self.dec_optimizer = torch.optim.Adam(self.dblock_list.parameters(), lr=dec_parm['stepsize'])
        self.z_smpls = []

    def prior_sample(self, z0_shape):
        self.dblock_list.eval()
        self.dblock_list.requires_grad_(False)
        self.z_smpls = []
        # sample from prior layer
        dblock = self.dblock_list[0]
        z = dblock.sample(z0_shape)
        self.z_smpls.append(z)
        # sample from other latent blocks
        for dblock in self.dblock_list[1:-1]:
            z = dblock.sample(z)
            self.z_smpls.append(z)

        # cprobs & sample the image
        dblock = self.dblock_list[-1]
        z = self.z_smpls[-1]
        z, mue = dblock.sample(z, verbose=True)
        self.z_smpls.append(z)

        return mue

    def encoder_outputs(self, x):
        enc_outputs = []
        mu = x
        for eblock in self.eblock_list:
            mu, sig = eblock.forward(mu)
            enc_outputs.append([mu, sig])

        return enc_outputs

    def posterior_sample(self, x, stats=False):
        # notice that this function samples the latent variables
        # and a new image
        self.dblock_list.eval()
        self.dblock_list.requires_grad_(False)
        self.eblock_list.eval()
        self.eblock_list.requires_grad_(False)
        self.z_smpls = []
        # apply encoder
        enc_outputs = self.encoder_outputs(x)
        enc_outputs.reverse()
        # sample latents
        # prior layer
        dblock = self.dblock_list[0]
        z = dblock.sample_shifted(enc_outputs[0][0], enc_outputs[0][1], stats=stats)
        self.z_smpls.append(z)
        # other latent layers
        for [mue, sige], dblock in zip(enc_outputs[1:], self.dblock_list[1:-1]):
            z = dblock.sample_shifted(z, mue, sige, stats=stats)
            self.z_smpls.append(z)

        # mue and sample image
        dblock = self.dblock_list[-1]
        z = self.z_smpls[-1]
        z, mue = dblock.sample(z, verbose=True)
        self.z_smpls.append(z)

        return mue
 
    def decoder_learn_step(self):
        self.dblock_list.train()
        self.dblock_list.requires_grad_(True)
        self.dec_optimizer.zero_grad()
        loss = torch.zeros(1).to(next(self.dblock_list.parameters()).device)
        for z_in, z_out, dblock in zip(self.z_smpls[:-1], self.z_smpls[1:], self.dblock_list[1:]):
            nll = dblock.neg_llik(z_in, z_out)
            loss = loss + nll
        loss.backward()
        self.dec_optimizer.step()
        
        return loss.detach()
    
    def encoder_learn_step(self):
        self.dblock_list.eval()
        self.dblock_list.requires_grad_(False)
        self.eblock_list.train()
        self.eblock_list.requires_grad_(True)
        self.enc_optimizer.zero_grad()
        # apply encoder
        enc_outputs = self.encoder_outputs(self.z_smpls[-1])
        enc_outputs.reverse()

        # learn steps
        #prior block
        dblock = self.dblock_list[0]
        loss = dblock.neg_llik_shifted(self.z_smpls[0], enc_outputs[0][0], enc_outputs[0][1])
        # other latent blocks
        for dblock, z_in, z_out, [mue, sige] in zip(self.dblock_list[1:-1], 
                                                    self.z_smpls[:-2], 
                                                    self.z_smpls[1:-1], 
                                                    enc_outputs[1:]):
            nll = dblock.neg_llik_shifted(z_in, z_out, mue, sige)
            loss = loss + nll
        loss.backward()
        self.enc_optimizer.step()
        return loss.detach()

    def save_stats(self, path):
        vdict = {}
        for i, block in enumerate(self.dblock_list[:-1]):
            vdict[str(i)] = block.kldivs_av.detach().cpu().numpy()
        np.savez_compressed(path, **vdict)       

    def init_weights(self):
        for dblock in self.dblock_list[1:]:
            dblock.init_weights()
        for eblock in self.eblock_list:
            eblock.init_weights()

