"""transformations.py: Module containing image transformation
This is module which contains custom transformations of features (images) following the tutorial:

https://pytorch.org/vision/stable/transforms.html#functional-transforms

"""
import torchvision.transforms as tv_trans
import torchvision.transforms.functional as TF
import random
import torch 
import torch.nn as nn
import kornia as K
from typing import List,Tuple,Union,Optional
from easydict import EasyDict


# List of transformations:
    # Gaussian Blur
    # Random Affine transform
    # Random Perspective
    # Gaussian Noise Channelwise
    # RandomApply

class RandomGaussianBlur(tv_trans.GaussianBlur):
    """Wrapper around GaussianBlur on multiple inputs:
        https://pytorch.org/vision/stable/generated/torchvision.transforms.GaussianBlur.html#torchvision.transforms.GaussianBlur
        https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#GaussianBlur
    """

    def forward(self,*args:List[torch.Tensor])->Tuple[torch.Tensor]:
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        outputs = [TF.gaussian_blur(x, self.kernel_size, [sigma, sigma]) for x in args]
        return outputs

class RandomAffine(tv_trans.RandomAffine):
    """Wrapper around RandomAffine on multiple inputs sharing same width and height (last two dims):
        https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomAffine.html#torchvision.transforms.RandomAffine
    """

    def forward(self,*args:List[torch.Tensor])->Tuple[torch.Tensor]:
        img = args[0]
        _, height, width = TF.get_dimensions(img)
        img_size = [width, height]  # flip for keeping BC on get_params call
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        outputs = []
        for img in args:
            channels, _, _ = TF.get_dimensions(img)
            fill = self.fill
            if isinstance(img, torch.Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * channels
                else:
                    fill = [float(f) for f in fill]
            
            outputs.append(TF.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center))

        
        return outputs 

class RandomAffineKornia(K.augmentation.RandomAffine):
    """Wrapper around RandomAffine on multiple inputs sharing same width and height (last two dims):
        https://kornia.readthedocs.io/en/v0.2.0/augmentation.html#kornia.augmentation.RandomAffine
        https://kornia.readthedocs.io/en/v0.2.0/_modules/kornia/augmentation/augmentation.html#RandomAffine
    """

    def forward(self,*args:List[torch.Tensor]):
        # create random transformation for the first time
        outputs = []
        outputs.append(super(RandomAffineKornia,self).forward(args[0]))
        params = self._params
        for i in range(1,len(args)):
            outputs.append(super(RandomAffineKornia,self).forward(args[i],params))
        
        return outputs

class RandomPerspective(tv_trans.RandomPerspective):
    """Wrapper around RandomPerspective on multiple inputs
        https://pytorch.org/vision/0.14/generated/torchvision.transforms.RandomPerspective.html#torchvision.transforms.RandomPerspective
        https://pytorch.org/vision/0.14/_modules/torchvision/transforms/transforms.html#RandomPerspective
    """

    def forward(self,*args:List[torch.Tensor])->Tuple[torch.Tensor]:
        
        # No transformation
        if torch.rand(1) >= self.p:
            return args
        
        # Same transformation for every image        
        img = args[0]
        _, height, width = TF.get_dimensions(img)
        startpoints, endpoints = self.get_params(width, height, self.distortion_scale)

        outputs = []
        for img in args:
            fill = self.fill
            channels, height, width = TF.get_dimensions(img)
            if isinstance(img, torch.Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * channels
                else:
                    fill = [float(f) for f in fill]

            
            outputs.append(TF.perspective(img, startpoints, endpoints, self.interpolation, fill))
        
        return outputs
    
class GaussianNoiseChannelwise(torch.nn.Module):
    """GaussianNoiseChannelwise transformation 
        This transformation adds a Gaussian Noise (\mu = 0) to each element of image 
        with variance specified at channel level: sigma_{channel}^2.
        The segmentation mask is not altered.
    """

    def __init__(self,sigma: Union[list,int])->None:
        """
        Args: 
            sigma Union(list,int): standard deviation of Gaussian distribution for each channel. It is shared across if int.
        """
    
        super().__init__()
        self.sigma = sigma
        self.noise = None
    
    def get_params(self):
        return self.sigma,self.noise

    def forward(self, *args:List[torch.Tensor]) -> Union[torch.Tensor,Tuple[torch.Tensor,torch.Tensor]]:
        
        if len(args) > 2:
            raise ValueError(f'Inputs should be image and/or mask instead of {args}')
        
        img = args[0]
        mask = args[1] if len(args) == 2 else None
        
        C = img.shape[1]
        sigma = self.sigma
        if isinstance(sigma,(int,float)):
            sigma = [float(sigma)] * C
        else:
            sigma = [float(s) for s in sigma]
        sigma = torch.Tensor(sigma)

        noise = torch.rand(img.shape)
        noise = torch.einsum('i,aijk->aijk',sigma,noise)
        
        self.noise = noise

        if mask is None:
            return img + noise
        else:
            return img + noise, mask

class Compose(tv_trans.Compose):
    """Wrapper around Compose on multiple inputs
        https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html
        https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#Compose
    """

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

class RandomApply(nn.Module):
    """Apply randomly a list of transformations with a given probability.

    Copied (and alternated!) from https://pytorch.org/vision/0.14/_modules/torchvision/transforms/transforms.html#RandomApply
    

    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:

        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, *args):
        if self.p < torch.rand(1):
            return args
        for t in self.transforms:
            args = t(*args)
        return args


    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    p={self.p}"
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


# custom transformations 

class MyAugmentation(nn.Module):
  def __init__(self):
    super(MyAugmentation, self).__init__()

    # we define and cache our transformations:
    self.k1 = GaussianNoiseChannelwise((0.15, 0.25, 0.25))
    self.k2 = K.augmentation.RandomGaussianBlur((3,3),sigma=(5.,1.),p=0.75)
    self.k3 = K.augmentation.RandomHorizontalFlip(p=0.75)
    self.k4 = K.augmentation.RandomAffine([-45., 45.], [0., 0.15], [0.5, 1.5], [0., 0.15])

    # we create lists designating the usage 
    self.apply_on_mask = nn.ModuleList([self.k2,self.k3,self.k4]) 
    self.apply_on_images = nn.ModuleList([self.k1,self.k2,self.k3,self.k4])
    self.to_be_inversed  = nn.ModuleList([self.k3,self.k4])
  
  def forward(self, img: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor,Optional[torch.Tensor]]:
    # 1. apply transformations on img
    for t in self.apply_on_images:
        img = t(img)

    # 2. apply transformations on mask if not None
    if mask is None:
        return img,None
        
    for t in self.apply_on_mask:
        mask = t(mask,t._params) # keep same params
    
    return img, mask
  
  def inverse_last_transformation(self,input: torch.Tensor):
    """Retrive transformations from last forward apply and create inverse transformations"""
    
    transform_matrix = None
    for t in self.to_be_inversed:
        
        if transform_matrix is not None:
            transform_matrix = torch.einsum('ijk,ikl->ijl',t.transform_matrix,transform_matrix) # Socks'n'Shoes
        else:
            transform_matrix = t.transform_matrix
     
    inverse_transform_matrix = K.geometry.transform.invert_affine_transform(transform_matrix[:,:-1,:]) # B x 2 x 3 (drop last row) 
    inverse_input = K.geometry.transform.warp_affine(input,inverse_transform_matrix,input.shape[-2:])

    # print(transform_matrix)
    # print(inverse_transform_matrix)
    # print(torch.einsum('ijk,ilk->ijl',inverse_transform_matrix,transform_matrix[:,:-1,:]))

    return inverse_input

