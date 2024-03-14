import torch
from dataset import PascalVOCSampler, ImageNetLocSampler, CUBSampler
from model import get_model
from config import cfg  # cfg.paths is a list of paths to the datasets
from classif.matching import MatchingClassifier, NCM
from tools import ResizeModulo
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from models.maskBlocks import SAM
from uuid import uuid4
from augment.augmentations import crop_mask, crop
from PIL import Image
import os
import wandb

from models.DSM_SAM import DSM_SAM
from model import get_sam_model, CachedSamPredictor
from segment_anything import SamPredictor
from models.deepSpectralMethods import DSM

def main_loc(cfg):
    dataset = cfg.dataset.lower() # imagenetloc, CUBloc, pascalVOC
    if dataset == "imagenetloc":
        sampler = ImageNetLocSampler(cfg)
    elif dataset == "cubloc":
        sampler = CUBSampler(cfg)
    elif dataset == "pascalvoc":
        sampler = PascalVOCSampler(cfg)
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(size="s",use_v2=False).to(device)
    
    assert cfg.setting.query   in ["whole", "AMG", "hierarchical", "filtered", "unfiltered"]
    assert cfg.setting.support in ["whole", "filtered", "unfiltered"]
    
    if cfg.setting.query == "AMG":
        print("Using AMG")
        amg = SAM("b")
        
    if cfg.setting.query == "hierarchical":
        print("Using hierarchical")
        dsm_model = DSM(model=model, # same model as the one used for the classification
                        n_eigenvectors=cfg.dsm.n_eigenvectors,
                        lambda_color=cfg.dsm.lambda_color)
        dsm_model.to(device)
        sam = get_sam_model(size="b").to(device)
        sam_model = CachedSamPredictor(sam_model = sam, 
                                    path_to_cache=os.path.join(cfg.sam_cache, "embeddings", cfg.dataset),
                                    json_cache=os.path.join(cfg.sam_cache, "embeddings", cfg.dataset, "cache.json"))
        hierarchical = DSM_SAM(dsm_model, sam_model, 
                            nms_thr=cfg.hierarchical.nms_thr,
                            area_thr=cfg.hierarchical.area_thr,
                            target_size=224*2,) 
    
    transforms = T.Compose([
            ResizeModulo(patch_size=16, target_size=224, tensor_out=True),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]) # imagenet mean and std
        ])
    
    L_acc = []
    if cfg.classifier == "matching":
        classifier = MatchingClassifier(seed=42)
    elif cfg.classifier == "ncm":
        classifier = NCM(seed=42)
    
    pbar = tqdm(range(cfg.n_runs), desc="Runs")
    
    for episode_idx in pbar:
        
        support_images, temp_support_labels, query_images, temp_query_labels, annotations = sampler(seed_classes=episode_idx, seed_images=episode_idx)
        filtered_annotations = sampler.filter_annotations(annotations, filter=True)
        unfiltered_annotations = sampler.filter_annotations(annotations, filter=False)
        
        support_augmented_imgs = []
        support_labels = []
        
        for i, img_path in enumerate(support_images):
            img = Image.open(img_path).convert("RGB")
            
            if cfg.setting.support == "whole":
                support_augmented_imgs += [img]
                labels = [(temp_support_labels[i], i) for j in range(1)]
                
            elif cfg.setting.support in ["filtered", "unfiltered"]:
                bboxes = filtered_annotations[img_path] if cfg.setting.support == "filtered" else unfiltered_annotations[img_path]
                support_augmented_imgs += [img]
                for bbox in bboxes:
                    if dataset == "imagenetloc":
                        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                    elif dataset == "cubloc":
                        bbox = [bbox[0], bbox[1], bbox[2], bbox[3]] # no modification
                    elif dataset == "pascalvoc":
                        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                
                    support_augmented_imgs += [crop(img,bbox,dezoom=cfg.dezoom)]
                labels = [(temp_support_labels[i], i) for j in range(len(bboxes)+1)]
                
            support_labels += labels
        
        query_augmented_imgs = []
        query_labels = []
        
        for i, img_path in enumerate(query_images):
            img = Image.open(img_path).convert("RGB")
            
            if cfg.setting.query == "whole":
                query_augmented_imgs += [img]
                labels = [(temp_query_labels[i], i) for j in range(1)]
            
            elif cfg.setting.query in ["filtered", "unfiltered"]:
                bboxes = filtered_annotations[img_path] if cfg.setting.query == "filtered" else unfiltered_annotations[img_path]
                query_augmented_imgs += [img]
                for bbox in bboxes:
                    if dataset == "imagenetloc":
                        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                    elif dataset == "cubloc":
                        bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
                    elif dataset == "pascalvoc":
                        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                        
                    query_augmented_imgs += [crop(img, bbox, dezoom=cfg.dezoom)]
                labels = [(temp_query_labels[i], i) for j in range(len(bboxes)+1)]
            
            elif cfg.setting.query == "AMG":
                resized_img = ResizeModulo(patch_size=16, target_size=224*2, tensor_out=False)(img)
                masks = amg.forward(img = resized_img)
                masks = [mask["segmentation"] for mask in masks if mask["area"] > mask["segmentation"].shape[0]*mask["segmentation"].shape[1]*0.05]
                query_augmented_imgs += [resized_img]
                query_augmented_imgs += [crop_mask(resized_img, mask, dezoom=cfg.dezoom) for mask in masks]
                labels = [(temp_query_labels[i], i) for j in range(len(masks)+1)]
                
            elif cfg.setting.query == "hierarchical":
                masks, _, resized_img = hierarchical.forward(img = img, 
                                                path_to_img=img_path,
                                                sample_per_map=cfg.hierarchical.sample_per_map,
                                                temperature=cfg.hierarchical.temperature)
                masks = masks.detach().cpu().numpy()
                query_augmented_imgs += [resized_img]
                query_augmented_imgs += [crop_mask(resized_img, mask, dezoom=cfg.dezoom) for mask in masks]
                labels = [(temp_query_labels[i], i) for j in range(len(masks)+1)]
                
            query_labels += labels
            
        support_augmented_imgs = [transforms(img).to(device) for img in support_augmented_imgs]
        query_augmented_imgs = [transforms(img).to(device) for img in query_augmented_imgs]
        
        support_tensor = torch.zeros((len(support_augmented_imgs), 384)) # size of the feature vector WARNING: hardcoded
        query_tensor = torch.zeros((len(query_augmented_imgs), 384))
        
        with torch.inference_mode():
            for i in range(len(support_augmented_imgs)):
                inputs = support_augmented_imgs[i].unsqueeze(0)
                outputs = model(inputs).squeeze(0)
                support_tensor[i] = outputs

            for i in range(len(query_augmented_imgs)):
                inputs = query_augmented_imgs[i].unsqueeze(0)
                outputs = model(inputs).squeeze(0)
                query_tensor[i] = outputs
        acc = classifier(support_tensor, query_tensor, support_labels, query_labels)
        L_acc.append(acc)
        pbar.set_description(f"Last: {round(acc,2)}, avg: {round(np.mean(L_acc),2)}")
        if cfg.wandb:
            wandb.log({"running_accuracy": acc,
                        "average_accuracy": np.mean(L_acc),
                       })
            
    print("Average accuracy: ", round(np.mean(L_acc),2), "std: ", round(np.std(L_acc),2))
    print("All accuracies: ", np.round(L_acc,2))   


                
if __name__ == "__main__":
    
    print("Config:", cfg.sampler)

    """
    python main.py -t loc -d [imagenetloc, cubloc, pascalvoc] -w -m "Test on imagenetloc with hierarchical sampling"
    """
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", "-t", type=str, default="baseline", help="baseline, main, hierarchical")
    parser.add_argument("--wandb", "-w", action="store_true", help="use wandb")
    parser.add_argument("--dataset", "-d", type=str, default="imagenet", help="imagenet, cub, caltech, food, cifarfs, fungi, flowers, pets")
    parser.add_argument("--seed", "-s", type=int, default=0, help="seed for the run")
    parser.add_argument("--message", "-m", type=str, default="", help="message for the run, only used with wandb")
    args = parser.parse_args()

    cfg["type"] = args.type
    cfg["dataset"] = args.dataset

    if args.wandb:
        wandb.login()
        wandb.init(project="procom-transformers", entity="procom", notes=args.message)
        cfg["wandb"] = True
        wandb.config.update(cfg)
    if args.type == "loc":
        main_loc(cfg)
    else:
        raise ValueError(f"Unknown type of experiment: {args.type}")