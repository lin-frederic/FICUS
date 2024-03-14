import torch

import matplotlib.pyplot as plt
# solve import error
import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path not in sys.path:sys.path.append(path)
from dataset import EpisodicSampler, FolderExplorer
from model import get_model, forward_dino_v1
from config import cfg
from PIL import Image
import torch
import numpy as np
import scipy
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
import cv2
from tqdm import tqdm
from tools import ResizeModulo
from dataset import PascalVOCSampler

from scipy.ndimage import center_of_mass # find the baricenter of the image

# adapted from https://github.com/lukemelas/deep-spectral-segmentation

def knn_affinity(image, n_neighbors=[20, 10], distance_weights=[2.0, 0.1]): 
    """Computes a KNN-based affinity matrix. Note that this function requires pymatting"""
    try:
        from pymatting.util.kdtree import knn
    except:
        raise ImportError(
            'Please install pymatting to compute KNN affinity matrices:\n'
            'pip3 install pymatting'
        )
    image = np.array(image)/255.0
    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    n = w * h

    x = np.tile(np.linspace(0, 1, w), h) 
    y = np.repeat(np.linspace(0, 1, h), w)

    i, j = [], []

    for k, distance_weight in zip(n_neighbors, distance_weights):
        f = np.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
            axis=1,
            out=np.zeros((n, 5), dtype=np.float32),
        )

        distances, neighbors = knn(f, f, k=k)

        i.append(np.repeat(np.arange(n), k))
        j.append(neighbors.flatten())

    ij = np.concatenate(i + j)
    ji = np.concatenate(j + i)
    coo_data = np.ones(2 * sum(n_neighbors) * n)

    # This is our affinity matrix
    W = scipy.sparse.csr_matrix((coo_data, (ij, ji)), (n, n))
    W = torch.tensor(W.todense(),dtype=torch.float32)
    return W

class DSM(nn.Module):
    def __init__(self, model=None, n_eigenvectors=5,lambda_color=10,device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        if model is None:
            print("Instantiating a new model: DINO with ViT-Small")
            self.model = get_model(size="s",use_v2=False)
        else:
            self.model = model
        self.n = n_eigenvectors
        self.lambda_color = lambda_color
        self.device = device
        self.downsampling_factor = 4 # match the downsampling factor of the paper with patch size 16
        self.has_map = False

    def forward(self, img):
        assert len(img.shape) == 4, "The input must be a batch of images" 

        h, w = img.shape[2], img.shape[3]
        h_map, w_map = min(img.shape[2]//16, 224//16), min(img.shape[3]//16, 224//16) # limit the size of the feature map to 224x224
        with torch.inference_mode():
            attentions = forward_dino_v1(self.model,img).squeeze(0)
            attentions = attentions[1:] #remove cls token, shape is (h_featmap*w_featmap, D)
            attentions = attentions.permute(1,0) # (D,h_featmap*w_featmap)

            attentions = attentions.reshape(attentions.shape[0],img.shape[2]//16,img.shape[3]//16).unsqueeze(0) # (1,D,h_featmap,w_featmap)
            attentions = nn.functional.interpolate(attentions,size=(self.downsampling_factor*h_map,self.downsampling_factor*w_map),mode="bilinear") # upscale the feature map to the original image size
            attentions = attentions.squeeze(0).reshape(attentions.shape[1],-1)
            attentions = attentions.permute(1,0) # (self.downsampling_factor*h_featmap*self.downsampling_factor*w_featmap,D)
            feature_similarity = (attentions @ attentions.T)/(torch.norm(attentions,dim=1).unsqueeze(1) @ torch.norm(attentions,dim=1).unsqueeze(0))
            
            # keep only the positive values
            feature_similarity = feature_similarity * (feature_similarity>0)
            #downscale the image to calculate the color affinity matrix, should be (self.downsampling_factor*h_featmap*self.downsampling_factor*w_featmap,self.downsampling_factor*h_featmap*self.downsampling_factor*w_featmap)
            img = nn.functional.interpolate(img,size=(self.downsampling_factor*h_map,self.downsampling_factor*w_map),mode="bilinear")
            img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            img = ((img-img.min())/(img.max()-img.min()))*255
            img = img.astype(np.uint8) 
            color_affinity = knn_affinity(img) # is a numpy array
            #color_affinity = torch.tensor(color_affinity,dtype=torch.float32)
            color_affinity = color_affinity.to(self.device)
            similarity = feature_similarity + self.lambda_color*color_affinity
            D = torch.diag(torch.sum(similarity,dim=1))
            # do not normalize the laplacian matrix because the eigenvalues are very small
            L = D - similarity # L is (self.downsampling_factor*h_featmap*self.downsampling_factor*w_featmap,self.downsampling_factor*h_featmap*self.downsampling_factor*w_featmap)
        L = L.detach().cpu() # faster to do it on cpu (non parallelizable)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        # do not keep the first eigenvalue and eigenvector because they are constant
        eigenvalues, eigenvectors = eigenvalues[1:], eigenvectors[:,1:] 
        # the eigenvectors basis might have opposite orientation, so we need to flip the eigenvectors
        for i in range(eigenvectors.shape[1]):
            # flip if the median is more than the center of the value range
            if torch.median(eigenvectors[:,i]) > (eigenvectors[:,i].max()+eigenvectors[:,i].min())/2:
                eigenvectors[:,i] = -eigenvectors[:,i]
        
        eigenvectors = eigenvectors[:,:self.n]
        eigenvectors = eigenvectors.reshape(self.downsampling_factor*h_map,self.downsampling_factor*w_map,self.n) # (self.downsampling_factor*h_map,self.downsampling_factor*w_map,self.n)
        eigenvectors = eigenvectors.permute(2,0,1) # (self.n,self.downsampling_factor*h_map,self.downsampling_factor*w_map)
        eigenvectors = eigenvectors.detach().cpu().numpy()

        temp = []
        for vector in eigenvectors:
            vector = ((vector-vector.min())/(vector.max()-vector.min()))*255 # normalize between 0 and 255 to have a grayscale image
         
            temp.append(cv2.resize(vector.astype(np.uint8), 
                                dsize=(w,h),
                                            interpolation=cv2.INTER_NEAREST)) # resize to original image size
            


        eigenvectors = np.stack(temp,axis=0) # (self.n,h,w)

        return eigenvectors
     
    def set_map(self, img):
        self.og_shape = img.shape
        self.map = self.forward(img) # (self.n,h,w)
        self.has_map = True
        return self.map

    def sample_from_maps(self, sample_per_map=1, temperature=255*0.1):

        def get_point(density_map, target_size):
            """Sample a point (1) from a density map to a target size (of the original image)"""
            assert abs(np.sum(density_map) - 1) < 1e-3, f"The density map must be normalized, got {np.sum(density_map)}"

            H,W = density_map.shape[0], density_map.shape[1]
            h,w = target_size[2], target_size[3]

            density = density_map.reshape(-1)

            density = density / np.sum(density) # normalize

            cum_density = np.cumsum(density, axis=0)

            u = np.random.rand()

            idx = np.searchsorted(cum_density, u)

            (y,x) = np.unravel_index(idx, (H,W))

            x = int(x * h / H)
            y = int(y * w / W)

            return (x,y)


        
        """Sample n_samples images from the eigenvectors maps"""
        assert self.has_map, "You need to call set_map before sampling"
        
        res = np.zeros((self.n, sample_per_map, 2), dtype=np.int32) # (n, sample_per_map, 2)

        for i in range(self.n):
            density_map = softmax_2d(self.map[i], temperature=temperature)
            for j in range(sample_per_map):
                res[i,j] = get_point(density_map, self.og_shape)
        
        return res

    
        
def softmax_2d(x, temperature=1.0):
    temp = x.reshape(-1) / temperature
    temp = temp - np.max(temp) # shift to avoid overflow
    exp_x = np.exp(temp) 
    softmax_x = exp_x / np.sum(exp_x)
    softmax_x = softmax_x.reshape(x.shape)
    return softmax_x


def main(mode):
    if not os.path.exists("temp_dsm"):
        os.mkdir("temp_dsm")

    if mode=="imagenet":
    
        folder_explorer = FolderExplorer(cfg.paths)
        paths = folder_explorer()
        sampler = EpisodicSampler(paths = paths,
                                n_query= cfg.sampler.n_queries,
                                n_ways = cfg.sampler.n_ways,
                                n_shot = 5,)
        
        # dataset
        episode = sampler()
        imagenet_sample = episode["imagenet"]
        # support set
        support_images = [image_path for classe in imagenet_sample for image_path in imagenet_sample[classe]["support"]] # default n_shot=1, n_ways=5
        
    elif mode=="pascal":
        sampler = PascalVOCSampler(cfg)
        support_images, _, _, _, _ = sampler()

        
    dsm = DSM(lambda_color=1.0)
    dsm.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dsm.to(device)


    for image_path in tqdm(support_images):
        image = Image.open(image_path).convert("RGB")
        image = ResizeModulo(patch_size=16, target_size=224*2, tensor_out=False)(image)
        image_tensor = transforms.ToTensor()(image)
        image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])(image_tensor)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        eigenvectors = dsm.set_map(image_tensor) # (n,h,w) (instead of forward(image_tensor)
        # save the plot of the eigenvectors side by side for each image and save it
        fig, axs = plt.subplots(3, eigenvectors.shape[0], figsize=(15,15))

        # eigenvectors        
        kernel_s = 3 

        for i,ax in enumerate(axs[0]):

            density = softmax_2d(eigenvectors[i], temperature=255)
            density =  scipy.signal.convolve2d(density, np.ones((kernel_s,kernel_s)), mode="same")

            density = density/np.max(density)


            ax.imshow(density,cmap="viridis")
            
            rescaled = softmax_2d(eigenvectors[i], temperature=0.1)

            baricenter = center_of_mass(rescaled)

            ax.scatter(baricenter[1],baricenter[0],c="red")
            ax.axis("off")

        # thresholded eigenvectors 
        for i,ax in enumerate(axs[1]):
            
            #temp = eigenvectors[i]
            s = 5
            temp = scipy.signal.convolve2d(eigenvectors[i], np.ones((s,s)), mode="same")
            temp = (temp - temp.min())/(temp.max()-temp.min())
            temp = (255 * temp).astype(np.uint8)

            thr, mask = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            ax.imshow(mask,cmap="viridis")
            ax.set_title(thr)
            ax.axis("off")

        # sample points on the original image
            
        sample_points = dsm.sample_from_maps(sample_per_map=10, temperature=255*0.07)
            
        for i,ax in enumerate(axs[2]):
            
            ax.imshow(image)
            for point in sample_points[i]:
                ax.scatter(point[0],point[1],c="red")


        plt.savefig(f"temp_dsm/eigenvectors_{image_path.split('/')[-1]}")
        plt.close()



if __name__ == "__main__":
    main("pascal")