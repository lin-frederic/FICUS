# Adapted from https://arxiv.org/pdf/2109.14279.pdf (Section 3.2)

import torch
from torch import nn
from model import get_model, forward_dino_v1, show_attn, get_seed_from_attn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from scipy.signal import convolve2d as convolve
from tqdm import tqdm
import argparse

from tools import unravel_index

class Lost(nn.Module):
    def __init__(self, model, alpha, k = 100):
        super().__init__()
        self.k = k # indicates the cardinality of the seed expansion set
        self.model = model # DINO model
        self.alpha = alpha # alpha = 1 means that the seed is the pixel with the lowest degree
                           # alpha = 0 means that the seed is the barycenter of the thresholded (otsu) attention map
        assert 0 <= self.alpha <= 1, "alpha must be between 0 and 1"
        self.model.eval()
        
    def forward(self, img):
        """
        Args:
        img : input image (tensor) of shape (1,3,H,W)
        
        Output:
        mask : mask of shape (1,H_d,W_d) indicating the pixels 
        that are part of the seed expansion set
        H_d and W_d are the dimensions of the patched image
        """
        H_d, W_d = img.shape[2]//16, img.shape[3]//16
        assert img.shape[2] % 16 == 0 and img.shape[3] % 16 == 0, "image dimensions must be divisible by 16"
        # run through the model
        with torch.inference_mode():
            out = forward_dino_v1(self.model, img).squeeze(0) 
            # remove cls token
            out = out[1:] # (H_d*W_d, D)
            attn = show_attn(self.model, img, is_v2=False)
        
        # get attention seed
        attn_seed = get_seed_from_attn(attn) # (2,)
        
        # compute similarity matrix, degree matrix
        similarity_matrix = torch.matmul(out, out.T)              # (H_d*W_d, H_d*W_d)
        

        degree_matrix = similarity_matrix>=0                      # (H_d*W_d, H_d*W_d)
        degree_matrix = degree_matrix.type(torch.int64)           # (H_d*W_d, H_d*W_d)
        # select seed with lowest degree
        seed_degree = torch.argmin(degree_matrix.sum(dim=0)) # or dim = 1, doesn't matter 
        # (without loss of generality, we make a choice here)

        
        # the seed is a convex combination of the attention seed and the seed with lowest degree
        
        # unravel coordinates
        seed_degree = unravel_index(seed_degree, (H_d,W_d)) # (2,)
        # attn seed is already in (y,x) format

        # compute seed
        seed_degree = seed_degree.to("cpu") 
        attn_seed = attn_seed.to("cpu") 
        
        seed = self.alpha*seed_degree + (1-self.alpha)*attn_seed # (2,)
        seed = torch.round(seed).type(torch.int64) # (2,)

        
        # convert seed to index
        seed = seed[0]*W_d + seed[1] # (1,)
        
        # expand seed set on similarity matrix
        degree_matrix[seed][seed] = 255
        set_seed = degree_matrix[seed]                            # (H_d*W_d,)

        # limit cardinality of seed expansion set to k
        ordered_set_seed = torch.argsort(similarity_matrix[seed], descending=True) # returns indices
        set_seed[ordered_set_seed[self.k:]] = 0                        # (H_d*W_d,)

        # box extraction algorithm
        for i in range(len(set_seed)):
            if set_seed[i] == 0:
                continue
            else:
                if torch.sum(similarity_matrix[i][set_seed>0]) > 0: 
                # if the sum of the similarities between the current pixel and the pixels in the seed expansion set is > 0
                    set_seed[i] = similarity_matrix[i][set_seed==1].sum() # set the pixel to the sum of the similarities
                else:
                    set_seed[i] = 0

        # normalize
        set_seed = (set_seed - set_seed.min()) / (set_seed.max() - set_seed.min())
        set_seed = set_seed.reshape(H_d, W_d).detach().cpu().numpy()
        set_seed = np.uint8(set_seed*255)

        return set_seed


def main(n, is_grid, seed):
    
    if seed > 0:
        np.random.seed(seed)
    
    res = 224
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(size="s",use_v2=False) # loads a DINOv1 model, size s
    model.to(device)
    
    k = 400 if is_grid else 100 # 4 images so 400 patches
    up = 2 # upscaling factor
    k = k*up**2 # logical to keep the same ratio patches/seed expansion set
    lost = Lost(model, alpha=0., k=k)

    from config import cfg
    
    for index in tqdm(range(n)):
    
        root = cfg.paths.imagenet
        folder = np.random.choice(os.listdir(root))
        path = os.path.join(root,folder)
        
        
        # concatenate 4 images (grid)
        if is_grid:
            blank = Image.new("RGB", (res*2,res*2))
        
            for i,img_path in enumerate(os.listdir(path)[:4]):
                img = Image.open(os.path.join(path,img_path)).convert("RGB")
                #img = T.CenterCrop(res)(img)
                img = T.Resize((res,res), antialias=True)(img)
                blank.paste(img, (res*(i%2), res*(i//2)))
                
            img = blank
        else:
            name = np.random.choice(os.listdir(path))
            img = Image.open(os.path.join(path,name)).convert("RGB")
            #img = T.CenterCrop(res)(img)
            img = T.Resize((res,res), antialias=True)(img)
            
        #img.save(f"temp/img_{index}.png")
        plt.figure(figsize=(10,5))
        
        
        w, h = img.size
        
        w, h = int(w*up), int(h*up)
        img = T.Resize((h//16*16,w//16*16), antialias=True)(img)
        img_t = T.ToTensor()(img).unsqueeze(0).to(device)
        
        out = lost(img_t)

        out = cv2.resize(out.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)

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

        plt.subplot(1,3,1)
        plt.imshow(img)
        plt.axis("off")
        plt.title("Image")

        plt.subplot(1,3,2)
        plt.imshow(out)
        plt.axis("off")
        plt.title("Lost output")

        plt.subplot(1,3,3)
        plt.imshow(mask)
        plt.axis("off")
        plt.title("Mask : largest connected component")        
        


        plt.savefig(f"temp/lost_{index}.png")

        plt.close()
        
        
        
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", "-n", type=int, default=5, help="Number of images to process")
    parser.add_argument("--grid", action=argparse.BooleanOptionalAction, default=False, help="Whether to process a grid of images")
    parser.add_argument("--seed", "-s", type=int, default=-1, help="Random seed, set to -1 for no seed")
    n = parser.parse_args().n
    is_grid = parser.parse_args().grid
    seed = parser.parse_args().seed
    main(n, is_grid, seed)