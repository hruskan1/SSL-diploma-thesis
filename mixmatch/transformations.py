"""transformations.py: Module containing image transformation
This is module which contains custom transformations of features (images) following the tutorial:

https://pytorch.org/vision/stable/transforms.html#functional-transforms

"""
import torchvision.transforms as tv_trans
import torchvision.transforms.functional as TF
import random
import torch 
from typing import List,Tuple,Union


# List of transformations:
    # Gaussian Blur
    # Random Affine transform
    # Random Perspective
    # Random Normalized Perturbations

class GaussianBlur(tv_trans.GaussianBlur):
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

    def forward(self, *args:List[torch.Tensor]) -> Tuple[torch.Tensor]:
        
        if len(args) != 2:
            raise ValueError(f'Inputs should be image and mask instead of {args}')
        
        img,mask = args
        channels, height, width = TF.get_dimensions(img)
        sigma = self.sigma
        if isinstance(sigma,(int,float)):
            sigma = [float(sigma)] * channels
        else:
            sigma = [float(s) for s in sigma]
        

        sigma = torch.Tensor(sigma)
        noise = torch.rand(channels, height, width)
        noise = torch.einsum('i,ijk->ijk',sigma,noise)
        

        return img + noise, mask


class RandomApply(torch.nn.Module):
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



    


    
