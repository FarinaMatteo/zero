import os
from data.datautils import build_dataset
from data.augs import get_transforms


def prepare_dataset(tta_module, set_id, num_views, resolution, dataset_mode):
    data_folder = os.getenv('DATASET_DIR', './datasets')
    data_transform = get_transforms(tta_module, resolution=resolution, num_views=num_views)
    val_dataset = build_dataset(set_id, data_transform, data_root=data_folder, mode=dataset_mode)
    return val_dataset
