from box import Box
from pprint import pprint

config = {
    "setting":{
    # can be any of ["whole", "AMG", "hierarchical", "filtered", "unfiltered"]
      "query": "hierarchical",
      "support": "filtered",
    },
    "classifier" : "ncm", # ["matching", "ncm"]
    "k_matching" : "1NN", # "1NN" or "KNN"
    "wandb" : False,
    "paths" : {
        "imagenet" : "path/to/imagenet",
        "imagenetloc" : "path/to/imagenet_loc",
        "cub" : "path/to/cub",
        "pascalVOC" : "path/to/pascalVOC",
    },
    "sampler" : {
        "n_ways" : 5,
        
        "n_shots" : 5,
        "n_queries" : 15,
    },
    
    "batch_size" : 32,
    "n_runs" : 100,

    "top_k_masks": 2, # top k masks used for each method
    "sam_cache" : "path/to/sam_cache", # path of imgs for which masks have been computed,
    "dataset": "not_specified",

    "dsm": {
        "n_eigenvectors" : 5, # number of eigenvectors to use for DSM
        "lambda_color" : 1 # as in the paper
    },

    "hierarchical": {
        "nms_thr": 0.1, # threshold for non-maximum suppression (mask)
        "area_thr": 0.01, # under this area, the mask is discarded
        "sample_per_map":10, # number of points sampled from each map
        "temperature":255*0.1 # the maps are normalized to [0,1] and then multiplied by temperature
    },

    "dezoom" : 0.2, # dezoom factor for the crop of the image

    "sam":"path/to/sam", # path to the SAM model
}

cfg = Box(config)

if cfg.sampler.n_ways is None:
    cfg.sampler.n_ways = {}
    for k,v in cfg.paths.items():
        cfg.sampler.n_ways[k] = 5
elif isinstance(cfg.sampler.n_ways, int):
    n_ways = cfg.sampler.n_ways
    cfg.sampler.n_ways = {k:n_ways for k in cfg.paths.keys()}


if __name__ == "__main__":
    pprint(cfg)