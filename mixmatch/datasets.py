"""datasets.py: Module containing custom datasets
This module is containg custom Dataset classes which accepts transforms specified in 'transformations.py'.

https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from sklearn.model_selection import train_test_split

import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.datasets.folder import (
    IMG_EXTENSIONS,
    default_loader,
    has_file_allowed_extension,
)

from typing import Tuple,Optional,Callable,Any,Union

import transformations as custom_transforms


# Code retaken from  https://github.com/Jonas1312/pytorch-segmentation-dataset and 
# adapted according to https://pytorch.org/vision/stable/transforms.html#functional-transforms guide

class SegmentationDatasetUnlabeled(data.Dataset):
    def __init__(self, dir_images, transform=None, extensions=None):
        """A dataloader for unlabeled (non-segmented) datasets.
        Args:
            dir_images (string): images directory path
            transform (callable, optional): A function/transform that takes in
            a input x and returns a transformed version x_aug. Defaults to None.
            extensions (list, optional): A list of allowed extensions. Defaults to None.
        
        """
        super().__init__()
        self.dir_images = dir_images
        self.transform = transform
        self.extensions = extensions if extensions else IMG_EXTENSIONS
        self.extensions = tuple(x.lower() for x in self.extensions)

        self.img_names = self.make_dataset()

    def __getitem__(self, index)->torch.Tensor:
        img_name = self.img_names[index]
        img = default_loader(os.path.join(self.dir_images, img_name))
        img = self.transform(TF.to_tensor(img)) if self.transform is not None else TF.to_tensor(img)

        return img

    def __len__(self):
        return len(self.img_names)

    def make_dataset(self):
        img_names = sorted(os.listdir(self.dir_images))

        img_names = [
            x for x in img_names if has_file_allowed_extension(x, self.extensions)
        ]

        return img_names


class SegmentationDatasetLabeled(SegmentationDatasetUnlabeled):
    def __init__(self, dir_images, dir_masks, transform=None, extensions=None):
        """A dataloader for labeled (segmented) datasets. Each datapoint contains a mask besides the original image
        Args:
            dir_images (string): images directory path
            dir_masks (string): masks directory path
            transform (callable, optional): A function/transform that takes in
            a x and returns a transformed version x_aug. Defaults to None, but
            implicitly parsed into Tensor. The transformation is applied on both
            image and mask (Only if they share same size!)
            extensions (list, optional): A list of allowed extensions. Defaults to None.
        
        """
        self.dir_masks = dir_masks
        super().__init__(dir_images=dir_images,transform=transform,extensions=extensions)
            
        self.img_names, self.mask_names = self.make_dataset()

    def __getitem__(self, index)->Tuple[torch.Tensor]:
        """"""
        img_name = self.img_names[index]
        mask_name = self.mask_names[index]
        img = default_loader(os.path.join(self.dir_images, img_name))
        mask = default_loader(os.path.join(self.dir_masks, mask_name))

        assert img.size == mask.size

        if not self.transform:
            return TF.to_tensor(img),TF.to_tensor(mask)
        
        
        img,mask = TF.to_tensor(img),TF.to_tensor(mask)
        print(img.shape)
        print(mask.shape)
        transformed = self.transform(torch.cat([img.unsqueeze(0),mask.unsqueeze(0)],dim=0))
        img,mask = tuple([x.squeeze(0) for x in torch.split(transformed,1,dim=0)])
        return img, mask


    def make_dataset(self):
        img_names = sorted(os.listdir(self.dir_images))
        mask_names = sorted(os.listdir(self.dir_masks))

        img_names = [
            x for x in img_names if has_file_allowed_extension(x, self.extensions)
        ]
        mask_names = [
            x for x in mask_names if has_file_allowed_extension(x, self.extensions)
        ]

        assert len(img_names) == len(mask_names)

        return img_names, mask_names


class CIFAR10Labeled(torchvision.datasets.CIFAR10):
    def __init__(self, 
                root:str,
                indicies:Optional[torch.Tensor]=None,
                train:bool=True,
                transform:Optional[Callable]=None,
                target_transform:Optional[Callable]=None,
                download:Optional[bool]=False)->None:
        """
        Module based on CIFAR 10 dataset https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html
        The data are kept as a numpy array
        The targets are parsed into the torch.tensor with dtype=long
        Args:
            indicies (Optional[torch.Tensor]): Indicies, which selects the subset of the original CIFAR10 dataset 
                (if None, the whole dataset is taken)
            rest: see original documentation
        """

        super(CIFAR10Labeled, self).__init__(root, train, transform, target_transform, download)

        self.targets = torch.Tensor(self.targets).to(torch.long)

        if indicies is not None:
            indicies = indicies.round().to(torch.long)
            self.data = self.data[indicies]
            self.targets = self.targets[indicies]


class CIFAR10Unlabeled(CIFAR10Labeled):
    def __init__(self, 
                root:str,
                indicies:Optional[torch.Tensor]=None,
                train:bool=True,
                transform:Optional[Callable]=None,
                target_transform:Optional[Callable]=None,
                download:Optional[bool]=False)->None:
        """
        Module based on CIFAR 10 dataset https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html
        The data are kept as a numpy array
        The targets are parsed into the torch.tensor with dtype=long (for debugging purposes)
        but are not obtained by __getitem__ method. 
        Args:
            indicies (Optional[torch.Tensor]): Indicies, which selects the subset of the original CIFAR10 dataset 
                (if None, the whole dataset is taken)
            rest: see original documentation
        """
        

        super(CIFAR10Unlabeled, self).__init__(root,indicies, train, transform, target_transform, download)

    
    def __getitem__(self, index: int) -> Any:
        """Returns transofrmed image"""
        img,_ = super().__getitem__(index)
        return img, None 

        
def train_test_val_split(arrays:list[Union[list,np.ndarray]],split:Tuple[float,float]=(0.2, 0.1), shuffle=True,random_state=None,stratify_index:Optional[int]=None):
    """Split each of arrays (dataset,labels) with ratio of  (1-split[0]-split[1]) / split[0] /split[1] 
    into train,test,val sets respectively. 
    Wrapper around https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    
    Args:
      arrays: list of indexables with same length / shape[0]
      split (Tuple[float,float]): tuple of split ratio in `test,val` order.
      stratify_index (int) : index which selects member of arrays according to which we stratify 
      (Therefore stratification mask has to be part of arrays)
      rest: See original documentation
      
    Returns:
      splitting: list, length= 3*len(arrays), with order: train,test,val.
    """
    stratify = arrays[stratify_index] if stratify_index is not None else None

    ret = train_test_split(*arrays, 
                           test_size=(split[0]+split[1]),
                           shuffle=shuffle,
                           random_state=random_state,
                           stratify=stratify)
    
    train_ = [ret[i] for i in range(0,len(ret),2)] # select the training / non-training part
    rest_  = [ret[i] for i in range(1,len(ret),2)]  
    
    stratify = rest_[stratify_index] if stratify_index is not None else None
    
    ret = train_test_split(*rest_,
                           test_size=(split[1] / (split[0]+split[1])),
                           shuffle=True,
                           random_state=random_state,
                           stratify=stratify)
    
    test_ = [ret[i] for i in range(0,len(ret),2)] 
    val_  = [ret[i] for i in  range(1,len(ret),2)] 
    return *train_, *test_, *val_ 

    

def get_CIFAR10(root:str,n_labeled:int,n_val:int,batch_size:int,download:bool,verbose:bool=False,onehot_flag=True)->Tuple[data.DataLoader,data.DataLoader,data.DataLoader,data.DataLoader]:
    """
    Creates dataloaders of CIFAR10 dataset with given parameters

    Args: 
        root (str): name of root directory of dataset
        n_labeled (int): Number of labeled examples to be extracted 
        n_val (int): Number of validation examples to be extracted (rest will be unlabeled)
        batch_size (int): Number of examples in one mini batch
        download (bool): If true, the dataset will be downloaded
        verbose (bool): If true print simple information about the splits of datasets
        one_hot_flag (bool): If true, apply onehot on the targets (THIS IS FOR DEBUGGING, REMOVE LATER)
    Returns:
        Tuple of (labeled_dataloader,unlabeled_dataloader,validation_dataloder,test_dataloder)
            validation_dataloder and test_dataloader is also labeled
            The features are transformed into the torch.Tensors and normalized
            The targets are one-hot encoded and returned as torch.Tensors(dtype=long)
    """
    # Taken from here https://stackoverflow.com/a/58748125/1983544
    import os
    num_workers = os.cpu_count() 
    if 'sched_getaffinity' in dir(os):
        num_workers = len(os.sched_getaffinity(0)) - 2
    num_workers = 0 # if on servers
    

    base_dataset = torchvision.datasets.CIFAR10(root,train=True,download=download,transform=transforms.ToTensor())
    t = base_dataset.targets

    # base_dataloader = data.DataLoader(base_dataset,batch_size,shuffle=True,num_workers=num_workers)
    # mean, std  = compute_mean_std(base_dataloader)
    mean = torch.Tensor([0.4914, 0.4822, 0.4465]) 
    std = torch.Tensor([0.2471, 0.2435, 0.2616])
    
    num_classes = 10
    
    _trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean,std=std)]) 
    if onehot_flag:
        _target_trans =transforms.Compose([custom_transforms.ToOneHot(num_classes)])
    else:
        _target_trans = None

    
    
    indicies_unlabeled, _,indicies_labeled,_,indicies_val,_= train_test_val_split([range(len(t)),t],
                                                                                    split = (n_labeled/len(t),n_val/len(t)),
                                                                                    shuffle=True,
                                                                                    random_state=None,
                                                                                    stratify_index=-1)
    
    labeled_cifar = CIFAR10Labeled(root=root,
                                   indicies = torch.Tensor(indicies_labeled).to(torch.long),
                                   train=True,
                                   transform=_trans,
                                   target_transform=_target_trans,
                                   download=False,
                                    )
    
    unlabeled_cifar = CIFAR10Labeled(root=root,
                                       indicies = torch.Tensor(indicies_unlabeled).to(torch.long),
                                       train=True,
                                       transform=_trans,
                                       target_transform=_target_trans,
                                       download=False
                                        ) # TODO: DEBUG PURPOSES PUT IT BACK TO UNLABELED LATER
    val_cifar = CIFAR10Labeled(root=root,
                                indicies = torch.Tensor(indicies_val).to(torch.long),
                                train=True,
                                transform=_trans,
                                target_transform=_target_trans,
                                download=False
                                )
                     
    test_cifar = CIFAR10Labeled(root=root,
                                train=False,
                                transform = _trans,
                                target_transform =_target_trans,
                                download=False)
    
    labeled_dataloader = data.DataLoader(labeled_cifar,batch_size,shuffle=True,num_workers=num_workers)
    unlabeled_dataloader = data.DataLoader(unlabeled_cifar,batch_size,shuffle=True,num_workers=num_workers)
    validation_dataloader = data.DataLoader(val_cifar,batch_size,shuffle=False,num_workers=num_workers)
    test_dataloader = data.DataLoader(test_cifar,batch_size,shuffle=False,num_workers=num_workers)

    if verbose:
        num_classes = 10
        print("Class distributions in sets")
        for i in range(num_classes):
            print(f"{i=}",end="\t")
            for t in [labeled_cifar,unlabeled_cifar,val_cifar,test_cifar]:
                print(f"{np.mean(np.array(t.targets) == i):2f}",end="\t")
            print()

        print("With datasets sizes: (respectively)")
        print(len(labeled_cifar),len(unlabeled_cifar),len(val_cifar),len(test_cifar))

    return labeled_dataloader, unlabeled_dataloader, validation_dataloader,test_dataloader


def compute_mean_std(dataloader:torch.utils.data.DataLoader)-> tuple:
    """
    Compute std (Bessels correction) and mean for whole dataset in param: dataloader
    
    param: dataloader containing data Tensor of shape [B,CH,H,W]
    returns: tuple(mean:Tensor,std:Tensor), of size [CH,] having mean and std across the whole dataset and height and width 
    """
    summ = 0
    num_of_instances = 0
    for _i,(data,_target) in enumerate(dataloader):
        num_of_instances += data.shape[0]
        summ += torch.sum(data,dim=[0,2,3])
        

    num_of_pixels = num_of_instances * data.shape[2] * data.shape[3]
    mean = summ / num_of_pixels

    var = torch.zeros_like(mean)
    for _i,(data,_target) in enumerate(dataloader):
    
        data = data.view(data.shape[0],data.shape[1],-1)
        m = mean.reshape(1,data.shape[1],1)
        var += torch.sum( (data - m )**2, dim=[0,-1])
        

    std = torch.sqrt(var / (num_of_pixels - 1) )
    
    return mean,std