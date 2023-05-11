"""datasets.py: Module containing custom datasets
This module is containg custom Dataset classes which accepts transforms specified in 'transformations.py'.

https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from collections import namedtuple
from sklearn.model_selection import train_test_split

import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.datasets.folder import (
    IMG_EXTENSIONS,
    default_loader,
    has_file_allowed_extension,
)

from torch.utils.data.dataset import Dataset
from typing import Tuple,Optional,Callable,Any,Union,List
from PIL import Image
from . import transformations as custom_transforms


# Code retaken from  https://github.com/Jonas1312/pytorch-segmentation-dataset and 
# adapted according to https://pytorch.org/vision/stable/transforms.html#functional-transforms guide

class SegmentationDatasetUnlabeled(Dataset):
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

    mean = torch.Tensor([0.4914, 0.4822, 0.4465]) 
    std = torch.Tensor([0.2471, 0.2435, 0.2616])

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


class CityScapeDataset(torchvision.datasets.Cityscapes):
    """
        Module based on CityScapes dataset https://pytorch.org/vision/main/generated/torchvision.datasets.Cityscapes.html
        This Datasets alternates the classes labels and enables same diversity as https://github.com/mcordts/cityscapesScripts.
        but alternating the labels on run. Also provides function to color the segmentation by default colors.
        The Dataset assume, that the loaded segmentations contains id specified below.
    """

    # See https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    # You can change the t_id as you wish, (see original)
    CityscapesClass = namedtuple(
        "CityscapesClass",
        ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
    )

    classes = [
        #               name                   id  train_id   category    category_id   has_instances  ignore_in_eval           colors
        CityscapesClass("unlabeled",            0,        0,  "void",               0,          False,      True,    (  0,   0,   0)),
        CityscapesClass("ego vehicle",          1,        0,  "void",               0,          False,      True,    (  0,   0,   0)),
        CityscapesClass("rectification border", 2,        0,  "void",               0,          False,      True,    (  0,   0,   0)),
        CityscapesClass("out of roi",           3,        0,  "void",               0,          False,      True,    (  0,   0,   0)),
        CityscapesClass("static",               4,        0,  "void",               0,          False,      True,    (  0,   0,   0)),
        CityscapesClass("dynamic",              5,        0,  "void",               0,          False,      True,    (111,  74,   0)),
        CityscapesClass("ground",               6,        0,  "void",               0,          False,      True,    ( 81,   0,  81)),
        CityscapesClass("road",                 7,        1,  "flat",               1,          False,      False,   (128,  64, 128)),
        CityscapesClass("sidewalk",             8,        1,  "flat",               1,          False,      False,   (244,  35, 232)),
        CityscapesClass("parking",              9,        1,  "flat",               1,          False,      True,    (250, 170, 160)),
        CityscapesClass("rail track",           10,       1,  "flat",               1,          False,      True,    (230, 150, 140)),
        CityscapesClass("building",             11,       2,  "construction",       2,          False,      False,   ( 70,  70,  70)),
        CityscapesClass("wall",                 12,       2,  "construction",       2,          False,      False,   (102, 102, 156)),
        CityscapesClass("fence",                13,       2,  "construction",       2,          False,      False,   (190, 153, 153)),
        CityscapesClass("guard rail",           14,       2,  "construction",       2,          False,      True,    (180, 165, 180)),
        CityscapesClass("bridge",               15,       2,  "construction",       2,          False,      True,    (150, 100, 100)),
        CityscapesClass("tunnel",               16,       2, "construction",        2,          False,      True,    (150, 120,  90)),
        CityscapesClass("pole",                 17,       3, "object",              3,          False,      False,   (153, 153, 153)),
        CityscapesClass("polegroup",            18,       3, "object",              3,          False,      True,    (153, 153, 153)),
        CityscapesClass("traffic light",        19,       3, "object",              3,          False,      False,   (250, 170,  30)),
        CityscapesClass("traffic sign",         20,       3, "object",              3,          False,      False,   (220, 220,   0)),
        CityscapesClass("vegetation",           21,       4, "nature",              4,          False,      False,   (107, 142,  35)),
        CityscapesClass("terrain",              22,       4, "nature",              4,          False,      False,   (152, 251, 152)),
        CityscapesClass("sky",                  23,       5, "sky",                 5,          False,      False,   ( 70, 130, 180)),
        CityscapesClass("person",               24,       6, "human",               6,          True,       False,   (220,  20,  60)),
        CityscapesClass("rider",                25,       6, "human",               6,          True,       False,   (255,   0,   0)),
        CityscapesClass("car",                  26,       7, "vehicle",             7,          True,       False,   (  0,   0, 142)),
        CityscapesClass("truck",                27,       7, "vehicle",             7,          True,       False,   (  0,   0,  70)),
        CityscapesClass("bus",                  28,       7, "vehicle",             7,          True,       False,   (  0,  60, 100)),
        CityscapesClass("caravan",              29,       7, "vehicle",             7,          True,       True,    (  0,   0,  90)),
        CityscapesClass("trailer",              30,       7, "vehicle",             7,          True,       True,    (  0,   0, 110)),
        CityscapesClass("train",                31,       7, "vehicle",             7,          True,       False,   (  0,  80, 100)),
        CityscapesClass("motorcycle",           32,       7, "vehicle",             7,          True,       False,   (  0,   0, 230)),
        CityscapesClass("bicycle",              33,       7, "vehicle",             7,          True,       False,   (119,  11,  32)),
        CityscapesClass("license plate",        -1,       0,  "vehicle",            7,          False,      True,    (  0,   0, 142)),
    ]

    id2trainid = {label.id: label.train_id for label in classes}
    trainid2color = {label.train_id : label.color for label in reversed(classes)}
    trainid2names = {label.train_id : label.name for label in reversed(classes)}
    num_classes = len(trainid2color.keys()) #- 1 #remove one if license_plate has '-1' (not valid class)
    

    class_weights = [1] * num_classes
    class_weights[0] = 0 # ignore 0 class (unlabeled)

    # std & mean for mode == fine 
    mean = torch.Tensor([0.485, 0.456, 0.406]) 
    std = torch.Tensor([0.229, 0.224, 0.225])

    def __init__( 
        self,
        root: str,
        split: str = "train",
        mode: str = "fine",
        target_type: Union[List[str], str] = "instance",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        

        super(CityScapeDataset, self).__init__(root,split,mode,target_type,transform,target_transform,transforms)

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Taken from https://pytorch.org/vision/main/_modules/torchvision/datasets/cityscapes.html#Cityscapes.__getitem__
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert("RGB")

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            elif t == 'semantic':
                # custom relabeling given by classes
                target = np.array(Image.open(self.targets[index][i]))
                for id,train_id in CityScapeDataset.id2trainid.items():
                    target[target==id] = train_id
        
                # to trick torch.ToTensor() !!!   
                target = target.astype('float')

            else:
                target = Image.open(self.targets[index][i])
                
            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target

    def get_original(self,index:int)-> Tuple[Any, Any]:
        """Get you the original segmentation from dataset"""
        return super(CityScapeDataset,self).__getitem__(index)
    
    @staticmethod
    def color_segmentation(segmentation:torch.Tensor,opacity:Optional[int]=None):
        """
        Parse a segmentation tensor into a color image tensor.

        Args:
            segmentation (torch.Tensor): a tensor of shape (N, 1, H, W) or (1, H, W)
                containing the segmentation labels.
            opacity: unsigned int [0,255]  seting the opacity of colors. 
        Uses:
            trainid2color (dict): a dictionary mapping train IDs (int) to color tuples (R, G, B).

        Returns:
            A tensor of shape (N, 3, H, W) or (3, H, W) containing the color image.
            If opacity is not None:
                A tensor of shape (N, 4, H, W) or (4, H, W) containing the color image.
        """

        # Get dimensions
        segmentation_dim = segmentation.ndim

        if segmentation_dim == 3:
            segmentation = segmentation.unsqueeze(0)
        
        N, _, H, W = segmentation.shape
        C = 4 if opacity is not None else 3
        # Create empty color image tensor
        
        color_img = torch.zeros((N, C, H, W), dtype=torch.float32).to(segmentation.device)
        

        # Loop over train IDs and fill color image tensor
        for trainid, color in CityScapeDataset.trainid2color.items():
            mask = (segmentation == trainid)
            if mask.any():
                if opacity is not None:
                    color_tensor = torch.tensor(color + (opacity,), dtype=torch.float32) / 255.0
                else:
                    color_tensor = torch.tensor(color, dtype=torch.float32) / 255.0
                    
                color_tensor = color_tensor.view(1, C, 1, 1).expand(N, -1, H, W).to(segmentation.device)
                mask = mask.expand(-1, C, -1, -1)
                color_img[mask] = color_tensor[mask]

        # Reshape to input shape 
        if segmentation_dim == 3:
            color_img = color_img.squeeze(0)

        return color_img
            
    @staticmethod
    def remove_normalization(normalized_inputs:torch.Tensor)->torch.Tensor:
        """
        Remove normalization (mean,std). Assumes torchvision.Normalization(std,mean) was applied.
        
        Args:
            normalized_inputs(torch.Tensor): batched images (N,C,H,W) 
        Returns
            original_inputs (torch.Tensor) (N,C,H,W)
        """
        
        N,C,H,W = normalized_inputs.shape
        std = CityScapeDataset.std.view(1, C, 1, 1).expand(N, -1, H, W).to(normalized_inputs.device)
        mean = CityScapeDataset.mean.view(1, C, 1, 1).expand(N, -1, H, W).to(normalized_inputs.device)
        original_inputs = (normalized_inputs * std) + mean 
        return original_inputs


class UnlabeledDataset(Dataset):
    
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self, index: int) -> Any:
        """Returns transofrmed image"""
        img,batched_output = self.dataset.__getitem__(index)
        return img, None * len(batched_output)
    
    def __len__(self):
        len(self.dataset)


def get_base_dataset(dl:data.DataLoader):
    """Gets base dataset from dataloader (if Subsey or ConcatDataset or others)"""
    
    # get original dataset
    dataset = dl.dataset
    while any(k == 'dataset' or k =='datasets' for k,v in dataset.__dict__.items()):
        try:
            dataset = dataset.dataset 
        except KeyError:
            dataset = dataset.datasets[0]
    return dataset

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

    
def get_CIFAR10(root:str,n_labeled:int,n_val:int,batch_size:int,download:bool,verbose:bool=False,onehot_flag=True)->Tuple[data.DataLoader,data.DataLoader,Optional[data.DataLoader],data.DataLoader]:
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
            validation_dataloder and test_dataloader is also labeled, if n_val == 0, validation_dataloder is None
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
    mean = CIFAR10Labeled.mean
    std = CIFAR10Labeled.std
    
    num_classes = 10
    
    _trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean,std=std)]) 
    if onehot_flag:
        _target_trans =transforms.Compose([custom_transforms.ToOneHot(num_classes)])
    else:
        _target_trans = None

    if n_val > 0:
        indicies_unlabeled, _,indicies_labeled,_,indicies_val,_= train_test_val_split([range(len(t)),t],
                                                                                        split = (n_labeled/len(t),n_val/len(t)),
                                                                                        shuffle=True,
                                                                                        random_state=None,
                                                                                        stratify_index=-1)
        
        val_cifar = CIFAR10Labeled(root=root,
                            indicies = torch.Tensor(indicies_val).to(torch.long),
                            train=True,
                            transform=_trans,
                            target_transform=_target_trans,
                            download=False
                            )
    
        validation_dataloader = data.DataLoader(val_cifar,batch_size,shuffle=False,num_workers=num_workers,)

    elif n_val == 0:
        indicies_unlabeled,indicies_labeled  = train_test_split(range(len(t)), 
                                                        test_size=n_labeled/len(t),
                                                        shuffle=True,
                                                        random_state=None,
                                                        stratify=t)
        validation_dataloader,val_cifar = None,None
    
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
                     
    test_cifar = CIFAR10Labeled(root=root,
                                train=False,
                                transform = _trans,
                                target_transform =_target_trans,
                                download=False)
    
    labeled_dataloader = data.DataLoader(labeled_cifar,batch_size,shuffle=True,num_workers=num_workers,drop_last=True)
    unlabeled_dataloader = data.DataLoader(unlabeled_cifar,batch_size,shuffle=True,num_workers=num_workers,drop_last=True)
    test_dataloader = data.DataLoader(test_cifar,batch_size,shuffle=False,num_workers=num_workers)

    if verbose:
        num_classes = 10
        print("Class distributions in sets")
        for i in range(num_classes):
            print(f"{i=}",end="\t")
            for t in [labeled_cifar,unlabeled_cifar,val_cifar,test_cifar]:
                if t is not None:
                    print(f"{np.mean(np.array(t.targets) == i):2f}",end="\t")
            print()

        print("With datasets sizes: (respectively)")
        if val_cifar is not None:
            print(len(labeled_cifar),len(unlabeled_cifar),len(val_cifar),len(test_cifar))
        else:
            print(len(labeled_cifar),len(unlabeled_cifar),len(test_cifar))

    return labeled_dataloader, unlabeled_dataloader, validation_dataloader,test_dataloader


def get_CityScape(root:str,n_labeled:int,n_val:int,batch_size:int,mode:str='fine',size:Tuple[int,int]=(128,256),target_type:str='semantic',verbose:bool=False,onehot_flag:bool=True):
    """
    Creates dataloaders of CityScapes dataset with given parameters. The normalization and rescaling 
    is applied so the final image has 'size' in spatial dimesions. The outputs are torch.Tenors. 

    Args: 
        root (str): name of root directory of dataset
        n_labeled (int): Number of labeled examples to be extracted 
        n_val (int): Number of validation examples to be extracted (rest will be unlabeled)
        batch_size (int): Number of examples in one mini batch
        mode(str): 'fine' or 'coarse' or 'all_coarse' if 'all_coarse', it merges 'train' and 'train_extra'. For else see
            https://www.cityscapes-dataset.com/examples/
        size(tuple(int,int)): Size of resized images
        target_type(str): 'instance', 'semantic', 'polygon', 'color', see 
            https://pytorch.org/vision/main/generated/torchvision.datasets.Cityscapes.html
        verbose (bool): If true print simple information about the splits of datasets
        one_hot_flag (bool): If true, apply onehot on the targets (THIS IS FOR DEBUGGING, REMOVE LATER)
    Returns:
        Tuple of (labeled_dataloader,unlabeled_dataloader,validation_dataloder,test_dataloder)
            validation_dataloder and test_dataloader is also labeled, if n_val == 0, validation_dataloder is None
            The features are transformed into the torch.Tensors and normalized
            The targets are one-hot encoded if onehot_flag == True and returned as torch.Tensors(dtype=long)
    """
    # Taken from here https://stackoverflow.com/a/58748125/1983544
    import os
    num_workers = os.cpu_count() 
    if 'sched_getaffinity' in dir(os):
        num_workers = len(os.sched_getaffinity(0)) - 2
    num_workers = 0 # if on servers
    
    # Mean and standard deviation for CityScapes
    mean = CityScapeDataset.mean
    std = CityScapeDataset.std
    num_classes = CityScapeDataset.num_classes 
    
    _trans = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean,std),
                             transforms.Resize(size,interpolation=InterpolationMode.BICUBIC)]) 
    
    
    if onehot_flag:
        _target_trans =transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize(size,interpolation=InterpolationMode.NEAREST),
                                           custom_transforms.ToOneHot(num_classes)])
    else:
        _target_trans = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Resize(size,interpolation=InterpolationMode.NEAREST)])

    if mode == 'all_coarse':
        datasets = []
        for split in ['train','train_extra']:
            datasets.append(CityScapeDataset(root = root,
                                            split = split,
                                            mode = mode,
                                            target_type = target_type,
                                            transform = _trans,
                                            target_transform = _target_trans)
                            )
                        
        base_dataset = data.ConcatDataset(datasets)
    else:
        base_dataset = CityScapeDataset(root,
                                        split = 'train',
                                        mode = mode,
                                        target_type = target_type,
                                        transform = _trans,
                                        target_transform = _target_trans)

    # if mode == 'fine', then sizes of  train/val/test is 2975/500/1525
    # if mode == 'corse' then sizes of train/val/train_extra is 2975/500/19998 
    num_examples = len(base_dataset)

    if n_labeled >= num_examples: 
        unlabeled_ = CityScapeDataset(root,split = 'train',mode = mode,target_type = target_type,transform = _trans,target_transform = _target_trans)
        labeled_ = CityScapeDataset(root,split = 'train',mode = mode,target_type = target_type,transform = _trans,target_transform = _target_trans)
        validation_dataloader,val_ = None,None # canot use n_val as they are dependent on training set
    else: # n_labeled < num_examples
        if n_val > 0:
            indicies_unlabeled,indicies_labeled,indicies_val = train_test_val_split([range(num_examples)],
                                                                                            split = (n_labeled/num_examples,n_val/num_examples),
                                                                                            shuffle=True,
                                                                                            random_state=None,
                                                                                            stratify_index=None)
            
            val_ = data.Subset(base_dataset,indicies_val)
        
            validation_dataloader = data.DataLoader(val_,batch_size,shuffle=False,num_workers=num_workers,)

        else: # n_val == 0
            indicies_unlabeled,indicies_labeled  = train_test_split(range(num_examples), 
                                                            test_size=n_labeled/num_examples,
                                                            shuffle=True,
                                                            random_state=None,
                                                            stratify=None)
            validation_dataloader,val_ = None,None
        
        unlabeled_ = data.Subset(base_dataset,indicies_unlabeled) 
        labeled_ = data.Subset(base_dataset,indicies_labeled)

    # unlabeled_ = UnlabeledDataset(unlabeled_) # TODO: DEBUG PURPOSES PUT IT BACK TO UNLABELED LATER
    test_ = CityScapeDataset(root,
                             split='val',
                             mode=mode,
                             target_type=target_type,
                             transform = _trans,
                             target_transform = _target_trans)
    
    unlabeled_dataloader = data.DataLoader(unlabeled_,batch_size,shuffle=True,num_workers=num_workers,drop_last=True)
    labeled_dataloader = data.DataLoader(labeled_,batch_size,shuffle=True,num_workers=num_workers,drop_last=True)
    test_dataloader = data.DataLoader(test_,batch_size,shuffle=False,num_workers=num_workers)

    if verbose:
        
        print("Datasets sizes: (respectively)")
        if val_ is not None:
            print(len(labeled_),len(unlabeled_),len(val_),len(test_))
        else:
            print(len(labeled_),len(unlabeled_),len(test_))
        print(f"With num classes: {num_classes}")

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