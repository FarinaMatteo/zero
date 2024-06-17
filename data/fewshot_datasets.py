import os
import json
import random
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

fewshot_datasets = [
    'DTD', 
    'Flower102', 
    'Food101', 
    'Cars', 
    'SUN397', 
    'Aircraft', 
    'Pets', 
    'Caltech101', 
    'UCF101', 
    'eurosat'
]

# dictionary of the format "dataset_id": ["image_dir", "json_split_file"]
path_dict = {
    "flower102": ["jpg", "split_zhou_OxfordFlowers.json"],
    "food101": ["images", "split_zhou_Food101.json"],
    "pets": ["images", "split_zhou_OxfordPets.json"],
    "dtd": ["images", "split_zhou_DescribableTextures.json"],
    "caltech101": ["images", "split_zhou_Caltech101.json"],
    "sun397": ["images", "split_zhou_SUN397.json"],
    "ucf101": ["images", "split_zhou_UCF101.json"],
    "cars": ["images", "split_zhou_StanfordCars.json"],
    "eurosat": ["images", "split_zhou_EuroSAT.json"]
}


class BaseJsonDataset(Dataset):
    def __init__(self, image_path, json_path, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []
        with open(self.split_json) as fp:
            splits = json.load(fp)
            samples = splits[self.mode]
            for s in samples:
                self.image_list.append(s[0])
                self.label_list.append(s[1])
    
        self.classes = set(self.label_list)

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
    
        return image, torch.tensor(label).long()


def build_fewshot_dataset(set_id, root, transform, mode='train', n_shot=None):
    if set_id.lower() == 'aircraft':
        return Aircraft(root, mode, n_shot, transform)
    
    path_suffix, json_path = path_dict[set_id.lower()]
    # dirname = os.path.dirname(json_path)
    # image_path = os.path.join(dirname, path_suffix)
    image_path = os.path.join(root, path_suffix)
    json_path = os.path.join(root, json_path)
    print("Building few-shot dataset with mode: ", mode)

    if set_id.lower() == 'cars':
        annot_file = os.path.join(root, "annots", "split_coop.csv")
        return BaseCsvDataset(image_path, annot_file, mode, n_shot, transform)
    else:
        return BaseJsonDataset(image_path, json_path, mode, n_shot, transform)


class Aircraft(Dataset):
    """ FGVC Aircraft dataset """
    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode

        self.cname = []
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))

        self.classes = set(self.label_list)

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()



class BaseCsvDataset(Dataset):
    def __init__(self, image_root, annot_file, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.image_root = image_root
        self.mode = mode
        self.annots = pd.read_csv(annot_file)
        self.annots = self.annots[self.annots['split'] == mode]
        self.labels = pd.read_csv(os.path.join(os.path.dirname(annot_file), 'labels.csv'))
        
        self.image_list = []
        self.label_list = []
        for idx, row in self.annots.iterrows():
            self.image_list.append(row['filename'])
            # find the label by looking up the label csv
            label = self.labels[self.labels['filename'] == row['filename']]['class_idx'].values[0]
            self.label_list.append(label)
            
        self.classes = set(self.label_list)

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_root, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label).long()
