"""
These blocks extract the masks from the image and return it.

- Identity : the mask is the whole image
- Lost : the masks are the seed expansion set + the whole image
- SAM : the masks are the masks from SAM Automatic Mask Generator
- DeepSpectralMethod : the masks are the thresholded eigenvectors of the Laplacian matrix
- Combined : the masks are the masks from SAM AMG + the masks from DeepSpectralMethod, optimized to be similar to the masks from Lost


Args: PIL image of shape (H,W,3) on CPU, we take H = W = 224
Out : list of {mask, area}
"""

from torch import nn
import torch
import numpy as np
from model import get_model
from torchvision import transforms as T
import cv2
from PIL import Image

from models.lost import Lost as Lost_module
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from models.deepSpectralMethods import DSM

from tools import iou, focal_loss, dice_loss

from config import cfg



class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, img):
        w, h = img.size
        mask = np.ones((w,h),dtype=np.uint8)
        mask[0,0] = 0 # remove one pixel to have a clean map
        return [{"segmentation": mask, "area": w*h}] # whole image

class Lost(nn.Module):
    def __init__(self,alpha, k, model = None):
        super().__init__()
        self.res = 224
        self.up = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is None:
            model = get_model(size="s",use_v2=False) # loads a DINOv1 model, size s
            model.to(self.device)
        self.lost = Lost_module(model=model,alpha=alpha,k=k*self.up**2)

    def clean(self, out, w, h):
        out = cv2.resize(out.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)
        # threshold the output to get a binary mask
        _, out = cv2.threshold(out, 128, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # get the largest connected component amongst the non-zero pixels

        components = cv2.connectedComponentsWithStats(out, connectivity=4)
        num_labels, labels, stats, centroids = components
        # identify the background label
        background_indexes = np.where(out == 0)
        background_label = np.median(labels[background_indexes])

        # sort the labels by area
        sorted_labels = np.argsort(stats[:, cv2.CC_STAT_AREA])[::-1] 
        # get the largest connected component
        largest_component_label = sorted_labels[0] if sorted_labels[0] != background_label else sorted_labels[1]

        # get the mask of the largest connected component
        mask = np.where(labels == largest_component_label, 1, 0).astype(np.uint8)

        return mask

    
    def forward(self, img):
        
        w, h = img.size

        new_w, new_h = int(w*self.up), int(h*self.up)

        img = T.Resize((new_h//16*16,new_w//16*16), antialias=True)(img)
        img = T.ToTensor()(img)
        img = img.unsqueeze(0).to(self.device)

        out = self.lost(img)
        mask = self.clean(out, w, h)

        return [{"segmentation": mask, "area": np.sum(mask)},]

class SAM(nn.Module):
    def __init__(self, size="b"):
        super().__init__()
        if size == "s":
            size = "b" # b is the smallest size
            print("Warning : SAM size s does not exist, using SAM size b instead")
        sizes = {
            "b" : "sam_vit_b_01ec64.pth",
            "l" : "sam_vit_l_0b3195.pth",
            "h" : "sam_vit_h_4b8939.pth",
        }

        sam = sam_model_registry[f"vit_{size}"](checkpoint=cfg.sam+sizes[size])
        print("SAM loaded")
        sam.to("cuda")

        self.AMG = SamAutomaticMaskGenerator(sam, 
                                             points_per_side=16,
                                             stability_score_thresh=0.82)
        
    def forward(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        masks = self.AMG.generate(img)
        masks = [{"segmentation": mask["segmentation"], 
                 "area": mask["area"]} for mask in masks]
        masks = sorted(masks, key=lambda x: x["area"], reverse=True) # sort by area (largest first)
        return masks

class DeepSpectralMethods(nn.Module):
    def __init__(self, model = None, n_eigenvectors=5, lambda_color=10):
        super().__init__()
        self. transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
        ])
        if model is None:
            model = get_model(size="s",use_v2=False)
        self.dsm = DSM(model=model, n_eigenvectors=n_eigenvectors, lambda_color=lambda_color)
    def clean(self, out, w, h):
        out = cv2.resize(out.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)
        # threshold the output to get a binary mask
        _, out = cv2.threshold(out, 128, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # get the largest connected component amongst the non-zero pixels

        components = cv2.connectedComponentsWithStats(out, connectivity=4)
        num_labels, labels, stats, centroids = components
        # identify the background label
        background_indexes = np.where(out == 0)
        background_label = np.median(labels[background_indexes])

        # sort the labels by area
        sorted_labels = np.argsort(stats[:, cv2.CC_STAT_AREA])[::-1] 
        # get the largest connected component
        largest_component_label = sorted_labels[0] if sorted_labels[0] != background_label else sorted_labels[1]

        # get the mask of the largest connected component
        mask = np.where(labels == largest_component_label, 1, 0).astype(np.uint8)

        return mask
    def forward(self, img):
        
        img = self.transforms(img).unsqueeze(0).to("cuda")
        eigenvectors = self.dsm(img) # returns a list of eigenvectors (arrays))

        masks = []

        for i in range(len(eigenvectors)):
            mask = eigenvectors[i]
            mask = cv2.resize(mask, (224,224), interpolation=cv2.INTER_NEAREST)
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            mask = np.where(mask > mask.mean(), 1, 0).astype(np.uint8)

            masks.append({"segmentation": mask, "area": np.sum(mask)})

        masks = sorted(masks, key=lambda x: x["area"], reverse=True) # sort by area
        for mask in masks:
            mask["segmentation"] = self.clean(mask["segmentation"], 224, 224)
        return masks


