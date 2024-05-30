#Machine learning model for bachelor project
#https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
#https://towardsdatascience.com/u-nets-with-resnet-encoders-and-cross-connections-d8ba94125a2c

from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import cv2
import torch
import numpy as np
from torchvision import transforms
import random
import torchvision.transforms.functional as F
import numbers
import torch.nn as nn
from typing import Sequence, Tuple, List, Optional
from torch import Tensor
from torchvision.transforms import _functional_pil as F_pil, _functional_tensor as F_t
from . import config
from PIL import Image, ImageOps
import warnings
import math


##############################################################
#This part is adapted from the their respective torch classes and functions to be able to handle random transforms using seeds.
def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)
    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)
    return size

def get_dimensions(img: Tensor) -> List[int]:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        if isinstance(img, torch.Tensor):
            return F_t.get_dimensions(img)

    return F_pil.get_dimensions(img)

class myRandomApply(torch.nn.Module):
    def __init__(self, transforms, seed, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p
        self.seed = seed

    def forward(self, img):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if self.p < torch.rand(1):
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    p={self.p}"
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class myRandomHorizontalFlip(torch.nn.Module):
    def __init__(self, seed, p=0.5):
        super().__init__()
        self.p = p
        self.seed = seed

    def forward(self, img):
        if torch.rand(1) < self.p:
            return F.hflip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

class myRandomVerticalFlip(torch.nn.Module):
    def __init__(self, seed, p=0.5):
        super().__init__()
        self.p = p
        self.seed = seed

    def forward(self, img):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if torch.rand(1) < self.p:
            return F.vflip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
    
class myRandomCrop(torch.nn.Module):

    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        _, h, w = F.get_dimensions(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __init__(self, size, seed, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        self.seed = seed

        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)
        

        return F.crop(img, i, j, h, w)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"

class MyRandomResizedCrop(torch.nn.Module):
    def __init__(
        self,
        size,
        seed,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.seed = seed
        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        if isinstance(interpolation, int):
            interpolation = F._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        _, height, width = F.get_dimensions(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", antialias={self.antialias})"
        return format_string
###############################################################

# Custom transform class for resizing images

class isLabel(torch.nn.Module):
    def __call__(self, img):
        if type(img) == torch.Tensor:
            # Resize images using bilinear interpolation
            resim = transforms.Resize(size=tuple(np.int16(np.ceil(random.uniform(0.5, 1.5) * np.array([650, 1250])))),
                                      interpolation=transforms.InterpolationMode.BILINEAR)(img)
        else:
            # Resize labels using nearest neighbor interpolation
            resim = transforms.Resize(size=tuple(np.int16(np.ceil(random.uniform(0.5, 1.5) * np.array([650, 1250])))),
                                    interpolation=transforms.InterpolationMode.NEAREST_EXACT)(img)
        return resim

class ResizeImageOrLabel:
    def __init__(self, target_size=(650, 1250), interpolation=transforms.InterpolationMode.BILINEAR):
        self.target_size = target_size
        self.interpolation = interpolation

    def __call__(self, img):
        # Resize images using bilinear interpolation
        if isinstance(img, torch.Tensor):
            target_size = tuple(np.int16(np.ceil(random.uniform(0.5, 1.5) * np.array(self.target_size))))
            res_img = transforms.Resize(size=target_size, interpolation=self.interpolation)(img)
        else:
            # Resize labels using nearest neighbor interpolation
            target_size = tuple(np.int16(np.ceil(random.uniform(0.5, 1.5) * np.array(self.target_size))))
            res_img = transforms.Resize(size=target_size, interpolation=transforms.InterpolationMode.NEAREST_EXACT)(img)
        return res_img


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, lossvalue=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.lossvalue = lossvalue

    def early_stop(self, validation_loss):
        if self.lossvalue is not None:
            if validation_loss <= self.lossvalue:
                return True
            return False
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Custom dataset class for segmentation
class SegmentationDataset(Dataset):

    # Store image, mask file paths, and augmentation transforms
    def __init__(self, imagePaths, labelPaths, T):
        self.imagePaths = imagePaths
        self.labelPaths = labelPaths
        self.T = T

    # Return the number of total samples contained in the dataset
    def __len__(self):
        return len(self.imagePaths)

    # Grab the image path from the current index and return a tuple of the image and its mask
    def __getitem__(self, idx):

        # Get the image corresponding to the current index
        imagePath = self.imagePaths[idx]
        labelPath = self.labelPaths[idx]

        # Read and convert the image to RGB
        image = Image.open(imagePath)
        label = Image.open(labelPath)

        # Convert image to tensor and set its type to float
        image = transforms.ToTensor()(image).type(torch.float)
        
        # Generate a random seed for data augmentation
        seed = np.random.randint(2147483647)

        # Define transformation pipeline
        transform = transforms.Compose([
            myRandomApply([ResizeImageOrLabel()], seed=seed, p=0.5),
            myRandomHorizontalFlip(seed=seed, p=0.5),
            myRandomVerticalFlip(seed=seed, p=0.5),
            myRandomCrop(seed=seed, size=config.patch_size[0])
        ])
       
        # Apply transformations based on the flag 'T'. T determines whether or not to use the full image as training input.
        if self.T is True:
            image = transform(image)
            label = transform(label)
            
        if self.T is False:
            transform = transforms.Compose([
            transforms.Resize(size=(325, 625)),
            myRandomHorizontalFlip(seed=seed, p=0.5),
            myRandomVerticalFlip(seed=seed, p=0.5),
            myRandomCrop(seed=seed, size=config.patch_size[0])
        ])
        
        # Squeeze the label tensor and set its type to int64
        label = torch.squeeze(transforms.PILToTensor()(label)).type(torch.LongTensor)
        
        return (image, label)