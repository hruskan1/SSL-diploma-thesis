"""datasets.py: Module containing custom datasets
This module is containg custom Dataset classes which accepts transforms specified in 'transformations.py'.

https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

"""

import os
import torch
from torch.utils import data

from torchvision import transforms
from torchvision.datasets.folder import (
    IMG_EXTENSIONS,
    default_loader,
    has_file_allowed_extension,
)
from torchvision.transforms import functional as TF
from typing import Tuple


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

        img = self.transform(TF.to_tesnor(img)) if self.transform else TF.to_tesnor(img)

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




