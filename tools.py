import torch
from typing import Tuple
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch.nn as nn
from PIL import Image

def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    """Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)


def iou(maskA, maskB):
    # maskA and maskB are (H,W) numpy arrays
    assert maskA.shape == maskB.shape
    # convert to boolean
    maskA = maskA>0
    maskB = maskB>0
    intersection = np.logical_and(maskA, maskB)
    union = np.logical_or(maskA, maskB)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def iou_torch(maskA, maskB):
    # maskA and maskB are (H,W) torch tensors
    assert maskA.shape == maskB.shape
    # convert to boolean
    maskA = (maskA>0).flatten()
    maskB = (maskB>0).flatten()
    intersection = torch.logical_and(maskA, maskB)
    union = torch.logical_or(maskA, maskB)
    iou_score = torch.sum(intersection) / (torch.sum(union)+1e-6)
    return iou_score

# 2 segmentation losses that could be used as a replacement for IoU

def focal_loss(inputs : np.ndarray,targets : np.ndarray, alpha : float = 1., gamma : float = 2.):
    # inputs and targets are numpy arrays of the same shape
    
    inputs = inputs.flatten()
    targets = targets.flatten()
    
    inputs = np.clip(inputs,1e-6,1-1e-6)
    targets = np.clip(targets,1e-6,1-1e-6)
    
    BCE = - (targets * np.log(inputs) + (1-targets) * np.log(1-inputs))
    
    BCE_EXP = np.exp(BCE)
    
    loss = np.mean(alpha * (1-BCE_EXP)**gamma * BCE)
    
    return loss

def dice_loss(inputs : np.ndarray,targets : np.ndarray, smooth : float = 1.):
    # inputs and targets are numpy arrays of the same shape
    
    inputs = inputs.flatten()
    targets = targets.flatten()

    
    intersection = np.sum(inputs * targets)
    union = np.sum(inputs) + np.sum(targets)
    
    loss = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - loss

def dice_torch(inputs : torch.Tensor,targets : torch.Tensor, smooth : float = 1.):
    # inputs and targets are torch tensors of the same shape    
    inputs = inputs.flatten()
    targets = targets.flatten()

    
    intersection = torch.sum(inputs * targets)
    union = torch.sum(inputs) + torch.sum(targets)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice

    

class PadAndResize(nn.Module):
    """
    Pad and resize an image to a target size
    Similar to SAM's ResizeLongestSide
    
    Parameters:
        target_size (int) : target size of the image (image will be squared)
        pad_value (int): value to pad the image with (between 0 and 255), default is 0
    Args:
        image (PIL image): image to pad and resize (RGB)
    Returns:
        tensor: tensor of shape (3, target_size, target_size)
    
    """
    def __init__(self, target_size, pad_value=0) -> None:
        assert isinstance(target_size, int) and target_size > 0, "target_size should be a positive int"
        assert isinstance(pad_value, int) and 0 <= pad_value <= 255, "pad_value should be an int between 0 and 255"
            
        self.target_size = target_size
        self.pad_value = pad_value
        self.to_tensor = T.ToTensor() # convert PIL image to tensor at the end (also normalize)
        
    def __call__(self, image):
        # image is a PIL image
        w, h = image.size
        
        new_w, new_h = self.target_size, self.target_size
            
        # resize
        max_size = max(w,h)
        ratio = max_size / self.target_size
        new_w = int(w / ratio)
        new_h = int(h / ratio)
        
        image = F.resize(image, (new_h,new_w))
        
        # pad
        
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        
        # compute padding
        left = delta_w // 2
        right = delta_w - left
        top = delta_h // 2
        bottom = delta_h - top
        
        image = F.pad(image, (left, top, right, bottom), self.pad_value)
        
        # convert to tensor
        image = self.to_tensor(image)
        
        return image
    
    
class ResizeModulo(nn.Module):
    """
    Resize an image to the multiple of the patch size closest to the target size
    
    Parameters:
        patch_size (int) : size of the patch (default is 16)
        target_size (int) : target size of the image (longest side)
    Args:
        image (PIL image): image to resize (RGB)
    Returns:
        tensor: tensor of shape (3, H, W) or image of size (H, W)
        where H and W are the closest multiples of patch_size and 
        the longest side is the closest to target_size
    """
    def __init__(self, patch_size=16, target_size=224, tensor_out=False) -> None:
        assert isinstance(patch_size, int) and patch_size > 0, "patch_size should be a positive int"
        assert isinstance(target_size, int) and target_size > 0, "target_size should be a positive int"
            
        self.patch_size = patch_size
        self.target_size = target_size
        self.tensor_out = tensor_out
        self.to_tensor = T.ToTensor()
    def __call__(self, image: Image,):
        # image is a PIL image
        w, h = image.size
        
        max_size = max(w,h)
        ratio = max_size / self.target_size
        new_w = int(w / ratio)
        new_h = int(h / ratio)
        
        # compute new size
        new_w = self.patch_size * (round(new_w / self.patch_size) + 1)
        new_h = self.patch_size * (round(new_h / self.patch_size) + 1)
        
        image = F.resize(image, (new_h,new_w))
        
        # convert to tensor
        if self.tensor_out:
            image = self.to_tensor(image)

        
        return image
    
    def apply_bbox(self, bbox, sizes):
        """
        Apply the transformation to a bbox (format [x,y,w,h]) from sizes (format [w,h])
        """

        w, h = sizes
        max_size = max(w,h)
        ratio = max_size / self.target_size
        new_w = int(w / ratio)
        new_h = int(h / ratio)

        # compute new size
        new_w = self.patch_size * (round(new_w / self.patch_size) + 1)
        new_h = self.patch_size * (round(new_h / self.patch_size) + 1)

        # apply the transformation
        bbox[0] = bbox[0] * new_w / w
        bbox[1] = bbox[1] * new_h / h
        bbox[2] = bbox[2] * new_w / w
        bbox[3] = bbox[3] * new_h / h

        bbox = list(map(int, bbox))

        return bbox


        


class preprocess_Features(torch.nn.Module):
    # preprocess_NCM: preprocess the features for NCM
    # including: 1. subtract the mean feature vector
    #            2. normalize the feature vector to make it into hypersphere
    # input: support_features: list of features, as a tensor
    #        query_features: list of features, as a tensor
    # output: support_output: list of features, as a tensor
    #         sphered_query_features: list of features, as a tensor
    def __init__(self):
        super(preprocess_Features, self).__init__()
    def forward(self, support_features, query_features):
        mean_feature = torch.mean(support_features, dim=0) # [d]
        sphered_support_features = support_features - mean_feature
        sphered_support_features = torch.nn.functional.normalize(sphered_support_features, p=2, dim=1)
        sphered_query_features = query_features - mean_feature
        sphered_query_features = torch.nn.functional.normalize(sphered_query_features, p=2, dim=1) # [n_query, d]
        
        return sphered_support_features, sphered_query_features