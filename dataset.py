import os
import random as rd
from typing import Any
from config import cfg
import json
import xml.etree.ElementTree as ET
class FolderExplorer():
    def __init__(self, dataset_paths) -> None:
        self.dataset_paths = dataset_paths
    
    def __call__(self) -> Any:
        # returns a dict where the key is the dataset name and the value is a dict of list
        
        dataset_dict = {}

        for dataset in self.dataset_paths:
            if dataset=="imagenet":
                dataset_dict["imagenet"]={}
                for class_name in os.listdir(cfg.paths["imagenet"]):
                    dataset_dict["imagenet"][class_name] = [os.path.join(cfg.paths["imagenet"], class_name, image_name) for image_name in os.listdir(os.path.join(cfg.paths["imagenet"], class_name))]
                    

            elif dataset=="cub":
                with open(os.path.join(cfg.paths["cub"], "train_test_split.txt")) as f:
                    train_test_split = f.readlines()
                    # filter train dataset with the image ids
                    test_dataset_image_ids = [line.split()[0] for line in train_test_split if line.split()[1]=="1"]
                    

                    test_dataset_images_ids_class = {}
                    
                with open(os.path.join(cfg.paths["cub"], "images.txt")) as f:
                    images = f.readlines()
                    # filter the train dataset with the image ids and class ids
                    for line in images :
                        if line.split()[0] in test_dataset_image_ids:
                            line = line.split()[1]
                            class_name= line.split('/')[0]
                            if class_name not in test_dataset_images_ids_class:
                                test_dataset_images_ids_class[class_name] = []
                            test_dataset_images_ids_class[class_name].append(os.path.join(cfg.paths["cub"], "images", line))

                dataset_dict["cub"] = test_dataset_images_ids_class # {class: [image_id, ...]}

            elif dataset=="cifarfs":
                dataset_dict["cifarfs"] = {}
                # folder structure: path_to_cifarfs/(meta_train/meta_test)/class_name/image_name
                metas = ["meta-test", "meta-val"]
                class_names = []
                for meta in metas:
                    for class_name in os.listdir(os.path.join(cfg.paths["cifarfs"], meta)):
                        class_names.append(os.path.join(meta, class_name))
                for class_name in class_names:
                    image_names = os.listdir(os.path.join(cfg.paths["cifarfs"], class_name))
                    dataset_dict["cifarfs"][class_name] = [os.path.join(cfg.paths["cifarfs"], class_name, image_name) for image_name in image_names]
            elif dataset=="fungi":
                dataset_dict["fungi"] = {}
                for class_name in os.listdir(os.path.join(cfg.paths["fungi"], "images")):
                    image_names = os.listdir(os.path.join(cfg.paths["fungi"], "images", class_name))
                    dataset_dict["fungi"][class_name] = [os.path.join(cfg.paths["fungi"], "images", class_name, image_name) for image_name in image_names]

            elif dataset=="caltech":
                dataset_dict["caltech"] = {}
                for class_name in os.listdir(os.path.join(cfg.paths["caltech"],"101_ObjectCategories")):
                    dataset_dict["caltech"][class_name] = os.listdir(os.path.join(cfg.paths["caltech"],"101_ObjectCategories", class_name))
                    dataset_dict["caltech"][class_name] = [img for img in dataset_dict["caltech"][class_name] if img.lower().endswith(".jpg") or img.lower().endswith(".png") or img.lower().endswith(".jpeg")]
                    dataset_dict["caltech"][class_name] = [os.path.join(cfg.paths["caltech"],"101_ObjectCategories", class_name, img) for img in dataset_dict["caltech"][class_name]]
            elif dataset=="food":
                dataset_dict["food"] = {}
                with open(os.path.join(cfg.paths["food"], "split_zhou_Food101.json"), "r") as split_file:
                    split_dict = json.load(split_file)
                    for item in split_dict["val"]:
                        image_name = item[0]
                        class_index = item[1]
                        if class_index not in dataset_dict["food"]:
                            dataset_dict["food"][class_index] = []
                        dataset_dict["food"][class_index].append(os.path.join(cfg.paths["food"], "images", image_name))
            elif dataset=="flowers":
                dataset_dict["flowers"] = {}
                with open(os.path.join(cfg.paths["flowers"], "split_zhou_OxfordFlowers.json"), "r") as split_file:
                    split_dict = json.load(split_file)
                    for item in split_dict["val"]:
                        image_name = item[0]
                        class_index = item[1]
                        if class_index not in dataset_dict["flowers"]:
                            dataset_dict["flowers"][class_index] = []
                        dataset_dict["flowers"][class_index].append(os.path.join(cfg.paths["flowers"], "jpg", image_name))
            elif dataset=="pets":
                dataset_dict["pets"] = {}
                with open(os.path.join(cfg.paths["pets"], "split_zhou_OxfordPets.json"), "r") as split_file:
                    split_dict = json.load(split_file)
                    for item in split_dict["val"]:
                        image_name = item[0]
                        class_index = item[1]
                        if class_index not in dataset_dict["pets"]:
                            dataset_dict["pets"][class_index] = []
                        dataset_dict["pets"][class_index].append(os.path.join(cfg.paths["pets"], "images", image_name))

                
            else:
                print(f"Dataset {dataset} not found")

        return dataset_dict

        
        

class EpisodicSampler():
    def __init__(self,paths, n_shot = 1,n_ways = {dataset:5 for dataset in cfg.paths.keys()}, n_query = 16) -> None:
        self.n_shot = n_shot
        self.n_query = n_query
        self.paths = paths
        self.n_ways = n_ways
        
    def __call__(self, seed_classes = None, seed_images = None) -> Any:
        """
        returns a dict where the key is the dataset name and the value is a dict of list

        seed_classes: int, seed for the random sampling of the classes
        seed_images: int, seed for the random sampling of the images

        E.g.: you want to sample from the same classes but different images, set seed_classes to a fixed value and seed_images to None
        """
        episode_dict = {}
        for dataset in self.paths:
            if seed_classes is not None:
                rd.seed(seed_classes)
            selected_classes = rd.sample(list(self.paths[dataset].keys()), self.n_ways[dataset])
            episode_dict[dataset] = {}
            for classe in selected_classes:
                episode_dict[dataset][classe] = {}
                if seed_images is not None:
                    rd.seed(seed_images)
                shuffle = rd.sample(self.paths[dataset][classe], min(self.n_shot+self.n_query, len(self.paths[dataset][classe])))
                episode_dict[dataset][classe]["support"] = shuffle[:self.n_shot]
                episode_dict[dataset][classe]["query"] = shuffle[self.n_shot:]
            
        return episode_dict
            
        
        
class DatasetBuilder():
    """
    return a dict where the key is the dataset name and the value is a tuple of 4 lists:
    support_images, support_labels, query_images, query_labels

    support_images: list of image paths
    support_labels: list of labels
    query_images: list of image paths
    query_labels: list of labels
    """
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        pass

    def __call__(self, seed_classes = None, seed_images = None) -> Any:
        folder_explorer = FolderExplorer(self.cfg.paths)
        paths = folder_explorer()
        sampler = EpisodicSampler(paths = paths,
                                n_query= self.cfg.sampler.n_queries,
                                n_ways = self.cfg.sampler.n_ways,
                                n_shot = self.cfg.sampler.n_shots)
        episode = sampler(seed_classes = seed_classes, seed_images = seed_images)
        # episode is (dataset, classe, support/query, image_path)
        dataset_dict = {}
        for dataset_name, list_classes in episode.items():
            support_images = [image_path for classe in list_classes for image_path in list_classes[classe]["support"]]
            support_labels = [classe for classe in list_classes for image_path in list_classes[classe]["support"]]
            query_images = [image_path for classe in list_classes for image_path in list_classes[classe]["query"]]
            query_labels = [classe for classe in list_classes for image_path in list_classes[classe]["query"]]
            dataset_dict[dataset_name] = (support_images, support_labels, query_images, query_labels)
        return dataset_dict
                
class COCOSampler():
    def __init__(self,cfg):
        self.path = cfg.paths["coco"]
        self.n_ways = 5
        self.n_shots = cfg.sampler.n_shots
        self.n_queries = cfg.sampler.n_queries
    def __call__(self, seed_classes = None, seed_images = None):
        with open(f"{self.path}/annotations/instances_val2017.json", "r") as f:
            data = json.load(f)
        all_images = [(data["images"][i]["id"], data["images"][i]["file_name"]) for i in range(len(data["images"]))]
        all_annotations = [(data["annotations"][i]["category_id"],
                            data["annotations"][i]["image_id"],
                            data["annotations"][i]["bbox"])
                            for i in range(len(data["annotations"]))]
        categories = {}
        # count the number of images per category
        # problem here: one image can be in multiple categories so redundancy
        for i in range(len(all_annotations)):
            if all_annotations[i][0] not in categories:
                categories[all_annotations[i][0]] = set()
            categories[all_annotations[i][0]].add(all_annotations[i][1])
        valid_categories = [] # we want only categories with sufficient images to sample
        for category in categories:
            if len(categories[category]) >= 4*(self.n_shots+self.n_queries): # 4*(k+m) to not always sample the same images
                valid_categories.append(category)
        if seed_classes is not None:
            rd.seed(seed_classes)
        selected_categories = rd.sample(valid_categories, self.n_ways)
        selected_images = {}
        seen_images = set()
        for category in selected_categories:
            selected_images[category] = []
            # take k+m images randomly from categories[category] but we need to make sure that
            # the images are not in seen_imgs to avoid duplicates among categories
            category_images = list(categories[category])
            if seed_images is not None:
                rd.seed(seed_images)   
            rd.shuffle(category_images)
            for img in category_images:
                if img not in seen_images:
                    selected_images[category].append(img)
                    seen_images.add(img)
                if len(selected_images[category]) == self.n_shots+self.n_queries:
                    break
            assert len(selected_images[category]) == self.n_shots+self.n_queries
        selected_annotations = {}
        for annotation in all_annotations:
            category_id, image_id, bbox = annotation
            if image_id in seen_images:
                if image_id not in selected_annotations:
                    selected_annotations[image_id] = []
                selected_annotations[image_id].append((category_id, bbox)) 
        for category in selected_categories:
            for img in selected_images[category]:
                selected_annotations[img] = (category, selected_annotations[img]) # (img_category, [(category, bbox), ...])
        for img in all_images:
            img_id, img_name = img
            if img_id in selected_annotations:
                img_path = f"{self.path}/images/val2017/{img_name}"
                selected_annotations[img_id] = (img_path, selected_annotations[img_id][0], selected_annotations[img_id][1])
        #selected_annotations = {img_id: (img_path, img_category, [(category, bbox), ...])}
        #group by category and split into support and query
        dataset = {}
        dataset["support"] = {}
        dataset["query"] = {}
        for img_id in selected_annotations:
            img_path, img_category, annotations = selected_annotations[img_id]
            if img_category not in dataset["support"]:
                dataset["support"][img_category] = []
                dataset["query"][img_category] = []
            if len(dataset["support"][img_category]) < self.n_shots:
                dataset["support"][img_category].append(selected_annotations[img_id])
            else:
                dataset["query"][img_category].append(selected_annotations[img_id])
        #dataset = {"support": {category: [(img_path, img_category, [(category, bbox), ...]), ...]}, "query": {category: [(img_path, img_category, [(category, bbox), ...]), ...]}
        return dataset
    
    def format(self, dataset):
        # format the dataset to:
        # (support_images, support_labels, query_images, query_labels, annotations)
        # annotations: {img_path: (img_category, [(category, bbox), ...])} (for all images)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        annotations = {}

        for category in dataset["support"]:
            for img in dataset["support"][category]:
                img_path, img_category, img_annotations = img
                support_images.append(img_path)
                support_labels.append(img_category)
                annotations[img_path] = (img_category, img_annotations)
        for category in dataset["query"]:
            for img in dataset["query"][category]:
                img_path, img_category, img_annotations = img
                query_images.append(img_path)
                query_labels.append(img_category)
                annotations[img_path] = (img_category, img_annotations)
        return support_images, support_labels, query_images, query_labels, annotations
    
    def filter_annotations(self, annotations, filter=True):
        # filter annotations to keep only the annotations that are of the same category as the image
        # annotations: {img_path: (img_category, [(category, bbox), ...])}
        filtered_annotations = {}
        for img_path in annotations:

            img_category, img_annotations = annotations[img_path]
            if filter:
                filtered_annotations[img_path] = [annotation[1] for annotation in img_annotations if annotation[0]==img_category]
            else:
                filtered_annotations[img_path] = [annotation[1] for annotation in img_annotations] # keep all annotations
        return filtered_annotations

class ImageNetLocSampler():
    def __init__(self, cfg):
        self.path = cfg.paths["imagenetloc"]
        self.n_ways = 5
        self.n_shots = cfg.sampler.n_shots
        self.n_queries = cfg.sampler.n_queries
        
    def __call__(self, seed_classes = None, seed_images = None):
        classes = os.listdir(cfg.paths["imagenet"])
        if seed_classes is not None:
            rd.seed(seed_classes)
        selected_classes = rd.sample(classes, self.n_ways)
        dataset = {}
        dataset["support"] = {}
        dataset["query"] = {}
        for classe in selected_classes:
            if seed_images is not None:
                rd.seed(seed_images)
            shuffle = rd.sample(os.listdir(os.path.join(cfg.paths["imagenet"], classe)), self.n_shots+self.n_queries)
            dataset["support"][classe] = shuffle[:self.n_shots]
            dataset["query"][classe] = shuffle[self.n_shots:]

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        annotations = {}

        full_annotations_dict = {}
        with open(f"{self.path}/LOC_val_solution.csv") as f:
            lines = f.readlines()
            for line in lines[1:]:
                # take the five first words 
                img, annotation_line = line.split(",")
                classe = annotation_line.split(" ")[0]
                # bbox are 4 words, it can be more than 1 bbox per image
                # now we want to list all the bboxes with x_max, x_min, y_max, y_min
                bboxes = []
                annotations_parsed = annotation_line.split(" ")[:-1]
                for i in range(0, len(annotations_parsed), 5):
                    bboxes.append((annotations_parsed[i], (int(annotations_parsed[i+1]), int(annotations_parsed[i+2]), int(annotations_parsed[i+3]), int(annotations_parsed[i+4]))))
                full_annotations_dict[img] = (classe, bboxes)
  

        for classe in dataset["support"]:
            for img in dataset["support"][classe]:
                img_path = os.path.join(cfg.paths["imagenet"], classe, img)
                support_images.append(img_path)
                support_labels.append(classe)
                img = img.split(".")[0]
                annotations[img_path] = full_annotations_dict[img]
        for classe in dataset["query"]:
            for img in dataset["query"][classe]:
                img_path = os.path.join(cfg.paths["imagenet"], classe, img)
                query_images.append(img_path)
                query_labels.append(classe)
                img = img.split(".")[0]
                annotations[img_path] = full_annotations_dict[img]

        return support_images, support_labels, query_images, query_labels, annotations  

    def filter_annotations(self, annotations, filter=True):
        filter_annotations = {}
        for img_path in annotations:
            img_category, img_annotations = annotations[img_path]
            if filter:
                filter_annotations[img_path] = [annotation[1] for annotation in img_annotations if annotation[0]==img_category]
            else:
                filter_annotations[img_path] = [annotation[1] for annotation in img_annotations]
        return filter_annotations    
    

class PascalVOCSampler():
    def __init__(self, cfg):
        self.path = cfg.paths["pascalVOC"]
        self.n_ways = cfg.sampler.n_ways["pascalVOC"]
        self.n_shots = cfg.sampler.n_shots
        self.n_queries = cfg.sampler.n_queries
    def __call__(self, seed_classes = None, seed_images = None):
        images = os.listdir(f"{self.path}/JPEGImages")
        annotations = os.listdir(f"{self.path}/Annotations")
        classes_trainval = os.listdir(f"{self.path}/ImageSets/Main")
        classes = set()
        for classe in classes_trainval:
            #strip the word after the underscore
            if classe not in ["train.txt","trainval.txt","val.txt"]:
                classes.add(classe.split("_")[0])
        #sample n_ways classes
        if seed_classes is not None:
            rd.seed(seed_classes)
            # because we want to be consistent with the seed, we convert the set to a list
            # then, we sort it because the hash of a set is not consistent
            classes = list(classes)
            classes.sort()
        selected_classes = rd.sample(classes, self.n_ways)
        dataset = {}
        dataset["support"] = {}
        dataset["query"] = {}
        for classe in selected_classes:
            classe_path = f"{self.path}/ImageSets/Main/{classe}_val.txt"
            with open(classe_path, "r") as f:
                images_classe = f.readlines()
                images_classe = [img.split()[0] for img in images_classe if img.split()[1]=="1"]
                if seed_images is not None:
                    rd.seed(seed_images)
                selected_images = rd.sample(images_classe, self.n_shots+self.n_queries)
                dataset["support"][classe] = selected_images[:self.n_shots]
                dataset["query"][classe] = selected_images[self.n_shots:]
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        annotations = {}
        for classe in dataset["support"]:
            for img in dataset["support"][classe]:
                img_path = f"{self.path}/JPEGImages/{img}.jpg"
                support_images.append(img_path)
                support_labels.append(classe)
                with open(f"{self.path}/Annotations/{img}.xml") as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    bboxes = []
                    for obj in root.findall("object"):
                        bbox = obj.find("bndbox")
                        bboxes.append((obj.find("name").text, (int(bbox.find("xmin").text), int(bbox.find("ymin").text), int(bbox.find("xmax").text), int(bbox.find("ymax").text))))
                    annotations[img_path] = (classe, bboxes)
        for classe in dataset["query"]:
            for img in dataset["query"][classe]:
                img_path = f"{self.path}/JPEGImages/{img}.jpg"
                query_images.append(img_path)
                query_labels.append(classe)
                with open(f"{self.path}/Annotations/{img}.xml") as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    bboxes = []
                    for obj in root.findall("object"):
                        bbox = obj.find("bndbox")
                        bboxes.append((obj.find("name").text, (int(bbox.find("xmin").text), int(bbox.find("ymin").text), int(bbox.find("xmax").text), int(bbox.find("ymax").text))))
                    annotations[img_path] = (classe, bboxes)
        return support_images, support_labels, query_images, query_labels, annotations  
    def filter_annotations(self, annotations, filter=True):
        # filter annotations to keep only the annotations that are of the same category as the image
        # annotations: {img_path: (img_category, [(category, bbox), ...])}
        filtered_annotations = {}
        for img_path in annotations:

            img_category, img_annotations = annotations[img_path]
            if filter:
                filtered_annotations[img_path] = [annotation[1] for annotation in img_annotations if annotation[0]==img_category]
            else:
                filtered_annotations[img_path] = [annotation[1] for annotation in img_annotations] # keep all annotations
        return filtered_annotations
class CUBSampler(): # with bbox
    def __init__(self, cfg):
        self.path = cfg.paths["cub"]
        self.n_ways = cfg.sampler.n_ways["cub"]
        self.n_shots = cfg.sampler.n_shots
        self.n_queries = cfg.sampler.n_queries
    def __call__(self, seed_classes = None, seed_images = None):
        # find all the classes
        with open(os.path.join(self.path, "classes.txt")) as f:
            classes = f.readlines()
            classes = [int(classe.split()[0]) for classe in classes] # get only the ordinal encoding of the class
        # find all the images and their classes
        with open(os.path.join(self.path,"image_class_labels.txt")) as f:
            image_class_labels = f.readlines()
        # sample n_ways classes
        if seed_classes is not None:
            rd.seed(seed_classes)
            # because we want to be consistent with the seed, we convert the set to a list
            # then, we sort it because the hash of a set is not consistent
            classes = list(classes)
            classes.sort()
        selected_classes = rd.sample(classes, self.n_ways)
        dataset = {}
        dataset["support"] = {}
        dataset["query"] = {}
        # find the images of the selected classes
        selected_images = {}
        for line in image_class_labels:
            img, classe = int(line.split()[0]), int(line.split()[1])
            if classe in selected_classes:
                if classe not in selected_images:
                    selected_images[classe] = []
                selected_images[classe].append(img)
        # sample n_shots+n_queries images for each class
        for classe in selected_images:
            if seed_images is not None:
                rd.seed(seed_images)
            selected_images[classe] = rd.sample(selected_images[classe], self.n_shots+self.n_queries)
            dataset["support"][classe] = selected_images[classe][:self.n_shots]
            dataset["query"][classe] = selected_images[classe][self.n_shots:]
        # find the images path
        with open(os.path.join(self.path, "images.txt")) as f:
            images = f.readlines()
            images = [(int(img.split()[0]), img.split()[1]) for img in images] # (img_id, img_path)
        with open(os.path.join(self.path, "bounding_boxes.txt")) as f:
            bboxes = f.readlines()
            bboxes = [(int(float(bbox.split()[0])), int(float(bbox.split()[1])), int(float(bbox.split()[2])), int(float(bbox.split()[3])), int(float(bbox.split()[4]))) for bbox in bboxes] # (img_id, x, y, width, height)
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        annotations = {}
        for type in ["support","query"]:
            for classe in dataset[type]:
                for img_id in dataset[type][classe]:
                    for (img_id2, img_path) in images:
                        if img_id == img_id2:
                            img_path_saved = os.path.join(self.path, "images", img_path)
                            if type == "support":
                                support_images.append(img_path_saved)
                                support_labels.append(classe)
                            else:
                                query_images.append(img_path_saved)
                                query_labels.append(classe)
                            for bbox in bboxes:
                                if bbox[0] == img_id:
                                    annotations[img_path_saved] = (classe,[(classe, bbox[1:])]) # (img_category, [(category, bbox), ...]), here we have only one bbox
        return support_images, support_labels, query_images, query_labels, annotations
    def filter_annotations(self, annotations, filter=True):
        # filter annotations to keep only the annotations that are of the same category as the image
        # annotations: {img_path: (img_category, [(category, bbox), ...])}
        filtered_annotations = {}
        for img_path in annotations:

            img_category, img_annotations = annotations[img_path]
            if filter:
                filtered_annotations[img_path] = [annotation[1] for annotation in img_annotations if annotation[0]==img_category]
            else:
                filtered_annotations[img_path] = [annotation[1] for annotation in img_annotations]
        return filtered_annotations
        
    
def main_pascal():
    pascal_sampler = PascalVOCSampler(cfg)
    support_images, support_labels, query_images, query_labels, annotations = pascal_sampler(seed_classes=1, seed_images=1)
    print(support_images)
    print(support_labels)
    #print(query_images)
    #print(query_labels)
    #print(annotations)
def main_imagenetloc():
    imagenetloc_sampler = ImageNetLocSampler(cfg)
    support_images, support_labels, query_images, query_labels, annotations = imagenetloc_sampler()
    print(support_images)
    print(support_labels)
    print(query_images)
    print(query_labels)
    print(annotations)
    
def main_cub():
    cub_sampler = CUBSampler(cfg)
    support_images, support_labels, query_images, query_labels, annotations = cub_sampler()
    print(support_images)
    print(support_labels)
    print(query_images)
    print(query_labels)
    print(annotations)

if __name__== "__main__":
    main_pascal()