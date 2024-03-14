import torch
from torch import nn
import cv2
from PIL import Image
import torchvision.transforms as T
import numpy as np
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor
import json
import os

from config import cfg

load_refs = {
    "s":"dinov2_vits14",
    "b":"dinov2_vitb14",
    "l":"dinov2_vitl14",
    "g":"dinov2_vitg14"
}

repo_ref = "facebookresearch/dinov2"


def get_model(size="s",use_v2=False):
    if use_v2:
        if size == "s":
            model = torch.hub.load(repo_ref, load_refs[size])
        elif size == "b":
            model = torch.hub.load(repo_ref, load_refs[size])
        elif size == "l":
            model = torch.hub.load(repo_ref, load_refs[size])
        elif size == "g":
            model = torch.hub.load(repo_ref, load_refs[size])
    else:
        if size == "s":
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        elif size == "b":
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        elif size == "l":
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitl16')
    return model

# adapted from the official repos
def forward_dino_v1(model, x):
    x = model.prepare_tokens(x)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    return x

# for dinov2, pass the is_training = True flag

# reimplementation of the SAM predictor to allow caching of the embeddings
class CachedSamPredictor(SamPredictor):
    def __init__(self,
                 path_to_cache: str,
                 json_cache: str,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_to_cache = path_to_cache
        self.json_cache = json_cache

        if not os.path.exists(self.path_to_cache):
            os.makedirs(self.path_to_cache)

        if not os.path.exists(self.json_cache):
            print(f"Creating json cache at {self.json_cache}")
            with open(self.json_cache, "w") as f:
                json.dump({}, f)

        with open(json_cache, "r") as f:
            self.cache = json.load(f)
        
    def set_image_cache(self, image_path, image):
        
        if "/" in image_path:
            image_name = "_".join(image_path.split("/")[-2:])
            # keep unique names (cf caltech dataset, where there are multiple images with the same name but in different folders)
        else:
            image_name = image_path


        if image_name in self.cache:
            x = torch.load(f"{self.cache[image_name]}.pt")
            self.original_size = x["original_size"]
            self.input_size = x["input_size"]
            self.features = x["features"]
            self.is_image_set = True
        else:
            if isinstance(image, Image.Image):
                image = np.array(image)
            elif isinstance(image, np.ndarray):
                pass
            else:
                raise ValueError(f"image must be a PIL image or a numpy array, got {type(image)}")
            
            self.set_image(image)
            to_save = {
                "original_size":self.original_size,
                "input_size":self.input_size,
                "features":self.features
            }
            torch.save(to_save, f"{os.path.join(self.path_to_cache, image_name)}.pt")
            self.cache[image_path] = os.path.join(self.path_to_cache, image_name)
            with open(self.json_cache, "w") as f:
                json.dump(self.cache, f)

def show_attn(model, x, is_v2=False):
    if is_v2:
        B, C, H, W = x.shape
        w_featmap = W // 14
        h_featmap = H // 14
        with torch.inference_mode():
            output = model.get_intermediate_layers(x = x,
                                                   reshape = True,
                                                   n = 2,
                                                   return_class_token = True,)
            maps = output[0][0] 
            B, C = output[0][1].shape
            
            # reshape maps to be (B, N, C) where N is the number of patches
            maps = maps.reshape((B,maps.shape[1],-1)).permute(0,2,1)
            class_token = output[0][1].reshape((B,-1,1)).permute(0,2,1)
            maps = torch.cat((class_token, maps), dim=1)
            # get the last attention block (only qkv)with 
            qkv = model.blocks[-1].attn.qkv
            B, N, C = maps.shape
            qkv_out = qkv(maps).reshape(B, N, 3, model.num_heads, C // model.num_heads).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, C//num_heads)
            head_dim = C // model.num_heads
            scale = head_dim**-0.5
            q, k = qkv_out[0] * scale, qkv_out[1]
            attn = q @ k.transpose(-2, -1) # (B, nh, N, N)
            nh = model.num_heads
            assert B == 1, "B must be 1"
            attn = attn[:, :, 0, 1:].reshape(B,nh, h_featmap, w_featmap)
            return attn
        
    else: # dinov1
        B, C, H, W = x.shape
        w_featmap = W // 16
        h_featmap = H // 16
        attn_map = model.get_last_selfattention(x)
        attn_map = attn_map[:, :, 0, 1:].reshape(B, attn_map.shape[1], h_featmap, w_featmap)
        return attn_map
        
def get_seed_from_attn(attn_map):
    # attn_map is (B, nh, H, W)
    # size is (H, W) or S
    
    array_map = torch.min(attn_map, dim=1)[0].squeeze().detach().cpu().numpy()

    
    array_map = (array_map - array_map.min()) / (array_map.max() - array_map.min())
    array_map = (255*array_map).astype(np.uint8)
    
    _, array_map = cv2.threshold(array_map, int(0.75*255), 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    
    barycenter = center_of_mass(array_map)
    
    barycenter = (int(barycenter[0]), int(barycenter[1]))
    
    return torch.Tensor(np.ravel(barycenter))


def get_sam_model(size="b"):
    if size == "s":
        print("SAM model is not available for size s, using size b instead")
        size = "b"
        
    sizes = {
            "b" : "sam_vit_b_01ec64.pth",
            "l" : "sam_vit_l_0b3195.pth",
            "h" : "sam_vit_h_4b8939.pth",
        }

    print(f"Loading SAM model for size {size}")

    sam = sam_model_registry[f"vit_{size}"](checkpoint=f"{cfg.sam}/{sizes[size]}")
    return sam
    
def main():
    model = get_model(size="s",use_v2=False)
    
    img = Image.open("temp/img_8.png")
    
    img = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
    ])(img).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    img = img.to(device)
    
    with torch.inference_mode():
        attn_map = show_attn(model, img)
        
    array_map = torch.min(attn_map, dim=1)[0].squeeze().detach().cpu().numpy() 
    # min returns (values, indices)
    array_map = cv2.resize(array_map, (224,224))
    array_map = (array_map - array_map.min()) / (array_map.max() - array_map.min())
    array_map = (255*array_map).astype(np.uint8)
    
    _, array_map = cv2.threshold(array_map, 100, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    
    barycenter = center_of_mass(array_map)
    
    
    plt.imshow(array_map, cmap="viridis")
    
    plt.scatter(barycenter[1], barycenter[0], c="red", s=100, marker="x")
    
    plt.colorbar()
    
    plt.savefig("temp/attn.png")
    
    
        
    
if __name__ == "__main__":
    main()