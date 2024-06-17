from PIL import Image
from torchvision import transforms


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    NEAREST = InterpolationMode.NEAREST
except ImportError:
    BICUBIC = Image.BICUBIC
    NEAREST = Image.NEAREST


def convert_to_rgb(image):
    return image.convert("RGB")


def create_standard_clip_transforms(resolution=224):
    
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    base_transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=BICUBIC),
        transforms.CenterCrop(resolution)
    ])
    preprocess = transforms.Compose([
        convert_to_rgb,
        transforms.ToTensor(),
        normalize
    ])
    return base_transform, preprocess


def get_crop_and_flip(resolution):
    return transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
        ])
    

class MultiViewCropAndFlip:
    def __init__(self, base_transform, preprocess, crop_resolution=224, n_views=2):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.crop_resolution = crop_resolution
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        crop_and_flip = get_crop_and_flip(resolution=self.crop_resolution)
        views = [self.preprocess(crop_and_flip(x)) for _ in range(self.n_views)]
        return [image] + views
    

def get_transforms(tta_module, resolution=224, num_views=64):

    base_transform, preprocess = create_standard_clip_transforms(resolution=resolution)
    print(f"Using {tta_module.name} TTA module.")
    
    if tta_module.name.lower() in ("zero", "zero-rlcf"):
        data_transform = MultiViewCropAndFlip(
            base_transform, preprocess,
            crop_resolution=resolution,
            n_views=num_views-1
        )
    
    elif tta_module.name.lower() == "zeroshot":
        data_transform = transforms.Compose([
            base_transform,
            preprocess
        ])
    else:
        raise NotImplementedError(f"Transforms for {tta_module.name} not implemented.")

    return data_transform