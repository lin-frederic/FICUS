import augmentations as emb

from torchvision import transforms as T
from PIL import Image
from typing import Dict, List
import json

class mask2embedings():
    def __init__(self, 
                 masks:Dict,  # {path_to_image : [(mask 1 of the image, crop_bbox correponding to the mask), (mask 2, bbox) ...]}  
                 path_to_save:str,  
                 embedding:List[str], 
                 ): 
        
        self.emdeddings = embedding # list of embedding names desire 
        self.masks = masks # {img_path : [masks list with the best mask of the image, with their relative corp link to each mask]}

        # list of all images we have the path
        self.img_paths = list(self.masks.keys())
        self.images = {} # list of all the images 

        # compute of all the images 
        for path_image in self.img_paths:
            self.images[path_image] = (T.Resize((224,224), antialias=True)(Image.open(path_image).convert("RGB")))              

        # embedding of the images 
        self.emdedded_images = {}  # { path_img = {emdedding_name : [embdedingS]}} 

        # path to save the output 
        self.path_to_save = path_to_save

    def generate_embedding(self)->Dict:
        """ Generate the embdeings given masks , {img_name : {"embedding_mame" : emdedded image}}"""
        for path in self.img_paths: 
            image = self.images[path] 
            masks = self.masks[path]  # list of mask related to the image

            # get the embedding 
            embedname2embedimg = {} # {"embedding_mame" : emdedded image}

            for embedding in self.emdeddings: 
            # compute the choosen embedding (in the emb list) for each masks 

                # list with all the embedded images for each mask for the choosen embedding 
                embedded_masks = [] 
                if embedding =="gaussian noise":
                    for mask in range(len(masks)): 
                        emb_img = emb.gaussian_noise(image, mask[0]) # return an image
                        if len(mask)>1:
                            embedded_masks.append(emb.crop(emb_img, mask[1])) # mask[1] is the bbox
                        else : # no crop 
                            embedded_masks.append(emb_img)
                if embedding =="gaussian_blur":
                    for mask in masks: 
                        emb_img = emb.gaussian_blur(image, mask[0])
                        if len(mask)>1:
                            embedded_masks.append(emb.crop(emb_img, mask[1]))
                        else : 
                            embedded_masks.append(emb_img)
                if embedding =="remove_bg":
                    for mask in masks: 
                        emb_img = emb.remove_bg (image , mask[0])
                        if len(mask)>1:
                            embedded_masks.append(emb.crop(emb_img, mask[1]))
                        else : 
                            embedded_masks.append(emb_img)
                if embedding =="highlighted_contour":
                    for mask in masks: 
                        emb_img = emb.highlighted_contour(image, mask[0])
                        if len(mask)>1:
                            embedded_masks.append(emb.crop(emb_img, mask[1]))
                        else :  
                            embedded_masks.append(emb_img)

                if embedding =="attention": 
                    for mask in masks: 
                        embedded_masks.append()

                # add the embedded image to the names of the embeding in the global dict
                embedname2embedimg[embedding] = embedded_masks

            # add all the embedding to the image 
            self.emdedded_images[path] = embedname2embedimg 

        return self.emdedded_images


    def save_masks(self):
        """ the embeddings are save in a json file with their embedded image"""
        img2masks = self.generate_embedding ()

        with open(self.path_to_save , "w" ) as json_file: 
            json.dump(img2masks, json_file)
        print(f"Mask and images generated with {self.model} saved at {self.path_to_save}")

