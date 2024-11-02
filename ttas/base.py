import torch
import torch.nn as nn

from data.datautils import prepare_imagenet_classnames
from data.imagenet_prompts import imagenet_classes
from data.cls_to_names import (
    caltech101_classes, pets_classes, flower102_classes,
    food101_classes, aircraft_classes, dtd_classes, 
    cars_classes, sun397_classes, ucf101_classes, eurosat_classes
)



class BaseTTAModule(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.name = "BaseTTAModule"
        self.seed = kwargs.get("seed", "0")
    
    def prepare_optimizer(self):
        raise NotImplementedError
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __str__(self) -> str:
        return self.name
    
    @property
    def trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def prepare_for_training(self, set_id, arch):
        # imagenet variants
        if set_id in ['A', 'R', 'K', 'V', 'I']:
            classnames = prepare_imagenet_classnames(set_id, imagenet_classes)
        elif len(set_id) > 1: 
            classnames = eval("{}_classes".format(set_id.lower()))
        
        classnames = [name.replace("_", " ") for name in classnames]

        # pre-compute the text embeddings for all CLIPs around
        if hasattr(self.model, "reset_classnames"):
            self.model.reset_classnames(classnames)
        if hasattr(self, "reward_model") and hasattr(self.reward_model, "reset_classnames"):
            self.reward_model.reset_classnames(classnames)
        
        self.classnames = classnames



def confidence_filter(logits: torch.Tensor, probs: torch.Tensor, top:float, return_idx: bool=False):
    batch_entropy = -(probs * probs.log()).sum(1)
    full_idx = torch.argsort(batch_entropy, descending=False)
    filt_idx = full_idx[:int(batch_entropy.size()[0] * top)]
    if not return_idx:
        return logits[filt_idx]
    return logits[filt_idx], filt_idx, full_idx