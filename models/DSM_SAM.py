"""
Prompt SAM with coarse masks from DSM to segment images.
-> Hierarchical segmentation
"""

import sys
import os
from typing import Any
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path not in sys.path:sys.path.append(path)

from models.deepSpectralMethods import DSM
from segment_anything import SamPredictor # non cached version
from model import get_sam_model, CachedSamPredictor

from scipy.signal import convolve2d
from scipy.ndimage import binary_fill_holes

import torch

from torchvision import transforms as T
from tools import ResizeModulo, iou_torch, dice_torch
from PIL import Image
import cv2

from config import cfg
from dataset import PascalVOCSampler

import matplotlib.pyplot as plt
import numpy as np 

import time

from tqdm import tqdm

class DSM_SAM():
    def __init__(self, dsm_model: DSM, 
                 sam_model: CachedSamPredictor,
                 nms_thr=0.5,
                 area_thr=0.05, # under this threshold, the mask is discarded
                 target_size=224
                 ):
        super().__init__()
        self.dsm_model = dsm_model
        self.sam_predictor = sam_model
        self.transforms = T.Compose(
            [T.ToTensor(),  
             T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])
        
        self.dsm_transform = ResizeModulo(patch_size=16, target_size=target_size, tensor_out=False)

        self.nms_thr = nms_thr
        self.area_thr = area_thr

    def get_metric(self, ref_mask, pred_mask, metric="iou"):
        assert len(ref_mask.shape) == 2, "unbatch ref_mask"
        assert len(pred_mask.shape) == 2, "unbatch pred_mask"

        if abs(ref_mask.shape[0]/ref_mask.shape[1] - pred_mask.shape[0]/pred_mask.shape[1]) > 0.1:
            raise ValueError("ref_mask and pred_mask have different aspect ratios")
        
        
        if ref_mask.shape != pred_mask.shape:
            print("ref_mask and pred_mask have different shapes, resizing ref_mask to pred_mask shape")
            ref_mask = cv2.resize(ref_mask, pred_mask.shape[::-1]) # (W, H) -> (H, W)
        
        if metric == "iou":
            return iou_torch(ref_mask, pred_mask)
        elif metric == "dice":
            return 1-dice_torch(ref_mask, pred_mask)    
        else:
            raise NotImplementedError(f"Metric {metric} is not implemented")
        
    def get_coarse_mask(self, eigenvector, kernel_size=3, method="otsu"):

        if not eigenvector.dtype == np.uint8:
            eigenvector = eigenvector.astype(np.uint8)

        if method == "adaptive":
        
            temp = 255 - eigenvector
            mask = cv2.adaptiveThreshold(src=temp,
                                            maxValue=255,
                                            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            thresholdType=cv2.THRESH_BINARY,
                                            blockSize=kernel_size,
                                            C=2)
            mask = 255 - mask
            # conv2d 
            mask = convolve2d(mask, np.ones((kernel_size+2,kernel_size+2)), mode="same")
            

        elif method == "otsu":
            temp = convolve2d(eigenvector, np.ones((kernel_size,kernel_size)), mode="same")
            temp = ((temp / temp.max()) * 255).astype(np.uint8)
            _, mask = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        else :
            raise NotImplementedError(f"Method {method} is not implemented")

        mask = mask > 0
        mask = binary_fill_holes(mask)

        return mask


    def nms(self, masks, qualities, threshold, metric="iou"):
        # masks: (n_masks, H, W)
        # qualities: (n_masks)
        # threshold: threshold for NMS
        # metric: metric to use for NMS

        # sort masks by quality
        sorted_idx = torch.argsort(qualities, descending=True)

        # keep the best mask
        kept_masks = [masks[sorted_idx[0]]]
        kept_idx = [sorted_idx[0]]

        sorted_idx = sorted_idx[1:]

        for i in sorted_idx:
            mask = masks[i]
            # compute metrics
            ious = [self.get_metric(kept_mask, mask, metric=metric) for kept_mask in kept_masks]
            if all([iou < threshold for iou in ious]):
                kept_masks.append(mask)
                kept_idx.append(i)

        final_masks = []
        final_idx = []

        area_ratios = torch.Tensor([mask.sum()/(mask.shape[0]*mask.shape[1]) for mask in kept_masks])


        # at least one mask is kept : the one with the biggest area
        max_idx = torch.argmax(area_ratios).item()
        final_masks = [kept_masks[max_idx]]
        final_idx = [kept_idx[max_idx]]

        # filter based on area ratios
        for i, (mask, idx) in enumerate(zip(kept_masks, kept_idx)):
            if idx == final_idx[0]:
                # skip the already kept mask
                continue
            # compute metrics
            if area_ratios[i] > self.area_thr:
                final_masks.append(mask)
                final_idx.append(idx)
            # else, discard the mask
            
        return torch.stack(final_masks), torch.Tensor(final_idx).long()
        

    def forward(self, img, path_to_img, sample_per_map=10, temperature=255*0.1, use_cache=False):
        """
        Args:
        img : full size image
        path_to_img : path to the image (used for caching)
        sample_per_map : number of samples per eigen map
        temperature : temperature for sampling
        use_cache : if True, use the image cache

        Returns:
        final_masks : (n_masks, H, W)
        final_prompts : (n_masks, 1, 2)
        dsm_img : (H, W)

        Warning: the returned masks are not resized to the original image size but to the size used for DSM (ResizeModulo)
        
        """

        dsm_img = self.dsm_transform(img)

        w, h = dsm_img.size

        # resize to 1024 

        sam_img = T.Resize(1024)(dsm_img)

        # prepare for DSM
        img_tensor = self.transforms(dsm_img) 
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to("cuda")

        # compute eigen maps (will also be used as coarse masks)
        eigen_maps = self.dsm_model.set_map(img_tensor) # returns numpy array (n_eigen_maps, H, W)

        # compute embeddings for the resized image
        if use_cache:
            self.sam_predictor.set_image_cache(path_to_img, sam_img)
        else:
            if isinstance(img, Image.Image):
                self.sam_predictor.set_image(np.array(sam_img))
            elif isinstance(img, np.ndarray):
                self.sam_predictor.set_image(sam_img)
            else:
                raise TypeError(f"img must be a PIL image or a numpy array, got {type(sam_img)}")
        self.sam_predictor.original_size = (h, w)

        # sample points from eigen maps
        sample_points = self.dsm_model.sample_from_maps(sample_per_map=sample_per_map, temperature=temperature) # (n_eigen_maps, n_samples, 2)

        sample_points = sample_points.reshape(-1, 2) # (n_eigen_maps * n_samples, 2)

        sample_points = torch.from_numpy(sample_points).unsqueeze(1).to("cuda") # (n_eigen_maps * n_samples, 1, 2)

        tranformed_sample_points = self.sam_predictor.transform.apply_coords_torch(sample_points, original_size=(h,w)) # (n_eigen_maps * n_samples, 1, 2)
        points_labels = torch.ones(sample_points.shape[0]).unsqueeze(1).to("cuda") # (n_eigen_maps * n_samples, 1)

        # predict masks from sampled points
        masks, qualities, _ = self.sam_predictor.predict_torch(point_coords=tranformed_sample_points,
                                                                           point_labels=points_labels,
                                                                           multimask_output=True,)
        
        self.sam_predictor.reset_image() # no need to keep the image in memory

        # -> multimask_output sets the number of masks to 3 (3 granularity levels)
        # masks: (n_eigen_maps * n_samples, 3, H, W)
        # qualities: (n_eigen_maps * n_samples, 3)

        # filter the best mask among the 3 sizes
        kept_masks = []
        kept_qualities = []

        for i, (trimask, triquality) in enumerate(zip(masks, qualities)):
            # trimask: (3, H, W)
            idx = i//sample_per_map # eigen map index
     
            numpy_ref_mask = self.get_coarse_mask(eigen_maps[idx], kernel_size=3, method="otsu")

            coarse_ref_mask = torch.from_numpy(numpy_ref_mask).to("cuda") # (H, W)

            # compute metrics

            metrics = [self.get_metric(coarse_ref_mask, 
                                    mask, metric="iou") for mask in trimask]
            #[1:] 
            # 0 is the smallest mask, uncomment to discard it

            # keep the best one
            best_idx = torch.argmax(torch.Tensor(metrics)).item()
            kept_masks.append(trimask[best_idx])
            kept_qualities.append(triquality[best_idx])

        # NMS on the best masks
        kept_masks = torch.stack(kept_masks)
        kept_qualities = torch.stack(kept_qualities)            
        
        final_masks, final_indexes = self.nms(kept_masks, kept_qualities, threshold=self.nms_thr, metric="iou")
        final_prompts = sample_points[final_indexes]

        return final_masks, final_prompts, dsm_img

        
    
    def __call__(self, 
                 img, 
                 path_to_img,
                 sample_per_map=10, 
                 temperature=255*0.1,
                 use_cache=False):
        return self.forward(img, path_to_img, sample_per_map, temperature, use_cache)
    

def main(all_in_one=False, mode="pascal"):
    dsm_model = DSM(n_eigenvectors=5, 
                    lambda_color=1)
    dsm_model.to("cuda")

    sam = get_sam_model(size="b").to("cuda")

    sam_model = CachedSamPredictor(sam_model = sam, path_to_cache="temp/sam_cache", json_cache="temp/sam_cache.json")
    
    model = DSM_SAM(dsm_model, sam_model, nms_thr=0.1, area_thr=0.01, target_size=224*2)

    if mode == "pascal":
        cfg.sampler.n_ways = 20
        sampler = PascalVOCSampler(cfg)
        support_images, _, _, _, _ = sampler()

    else:
        type_ = "val"
        path = cfg.paths.pascalVOC+type_
        limit = 20

        support_images = [os.path.join(path, f) for f in os.listdir(path)]

        seed = np.random.randint(0, 1000) # 0 # 42 #
        
        print(f"Seed: {seed}")

        np.random.seed(seed)
        support_images = np.random.choice(support_images, limit)
        support_images = list(support_images)

    start = time.time()

    for img_path in tqdm(support_images):
        img = Image.open(img_path).convert("RGB")

        masks,points, resized_img = model(img,
                             img_path,
                             sample_per_map=10, 
                             temperature=255*0.1,)

        if all_in_one:
            # plot all masks in one figure (other one is the original image)

            plt.figure(figsize=(10, 5), dpi=200)
            plt.subplot(1, 2, 1)    
            plt.imshow(img) # could have better resolution
            plt.axis("off")

            plt.subplot(1, 2, 2)
            img_mask = np.zeros(masks[0].shape + (4,))
            img_mask[..., 3] = 0

            # order masks by area
            areas = torch.stack([mask.sum() for mask in masks]) 
            sorted_idx = torch.argsort(areas, descending=True)
            masks = [masks[i] for i in sorted_idx]
            points = [points[i] for i in sorted_idx]

            for mask, point in zip(masks, points):
                mask = mask.detach().cpu().numpy()
                m = mask
                color_mask = np.concatenate([np.random.random(3), [0.85]])
                img_mask[m] = color_mask
                plt.scatter(point[0,0].cpu(), point[0,1].cpu(), c="r", s=10)

            plt.imshow(resized_img)
            plt.imshow(img_mask)
            plt.axis("off")

        else:
            # square plot
            n = len(masks)+1
            h = int(np.sqrt(n))
            w = int(np.ceil(n/h))
            fig, axes = plt.subplots(h, w, figsize=(w*3, h*3))
            axes = axes.flatten()
            axes[0].imshow(img)
            axes[0].axis("off")

            for i, (mask, point) in enumerate(zip(masks, points)):
                ax = axes[i+1]
                ax.imshow(mask.detach().cpu().numpy(), cmap="Blues")
                ax.scatter(point[0,0].cpu(), point[0,1].cpu(), c="r", s=10)
                ax.axis("off")
            
            # fill the remaining axes
            for i in range(len(masks)+1, len(axes)):
                ax = axes[i]
                ax.imshow(resized_img)
                ax.axis("off")
            plt.tight_layout()

        plt.savefig(f'temp/{"_".join(img_path.split("/")[-2:])}.png')
        plt.close()
    end = time.time()
    print(f"Time: {end-start:.3f}s")


if __name__ == "__main__":
    main(all_in_one=True, mode="pascal")
    print("Done!")
