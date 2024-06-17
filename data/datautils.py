import os
import torchvision.datasets as datasets
from data.fewshot_datasets import build_fewshot_dataset, fewshot_datasets


ID_to_DIRNAME = {
    'I': 'imagenet',
    'A': 'imagenet-a',
    'K': 'imagenet-sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'Flower102': 'flower102',
    'DTD': 'dtd',
    'Pets': 'oxford_pets',
    'Cars': 'stanford_cars',
    'UCF101': 'ucf101',
    'Caltech101': 'caltech-101',
    'Food101': 'food101',
    'SUN397': 'sun397',
    'Aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    
    # natural distribution shifts datasets
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    
    # fine-grained classification datasets
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id]), transform, mode=mode)
    else:
        raise NotImplementedError
        
    return testset


def prepare_imagenet_classnames(set_id, imagenet_classes):
    assert set_id in ['A', 'R', 'K', 'V', 'I']
    from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

    classnames_all = imagenet_classes
    classnames = []
    if set_id in ['A', 'R', 'V']:
        label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
        if set_id == 'R':
            for i, m in enumerate(label_mask):
                if m:
                    classnames.append(classnames_all[i])
        else:
            classnames = [classnames_all[i] for i in label_mask]
    else:
        classnames = classnames_all
    return classnames
