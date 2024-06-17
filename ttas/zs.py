import os
import torch
from copy import deepcopy
from .base import BaseTTAModule
from utils.tools import print
from clip.custom_clip import get_clip

class ZeroShot(BaseTTAModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prepare_model(*args, **kwargs)
        self._freeze_params()
        self.name = "ZeroShot"
        

    def prepare_model(self, *args, **kwargs):
        if kwargs.get("model") == "clip":
            self.model = get_clip(
                kwargs.get("arch"),
                kwargs.get("gpu"),
                kwargs.get("ctx_init"),
                freeze_text=True,
                freeze_vision=True,
                cache_text_features=True,
                use_text_templates=bool(kwargs.get("use_templates")),
                maple_weights=kwargs.get("maple_weights"),
                maple_seed=kwargs.get("seed")
            )

        else:
            raise NotImplementedError(f"{kwargs.get('model')} not implemented.")
        print("=> Model created: visual backbone {}".format(kwargs.get("arch")))


    def _freeze_params(self):
        for name, param in self.model.named_parameters():
            # enables distribution for clip
            if "proj" not in name:
                param.requires_grad_(False)
            
            # enables distribution for resnet50
            if name.startswith("model.fc"):
                param.requires_grad_(True)
        print('=> Freezing all parameters.')
        return

    def reset(self):
        pass

    def prepare_optimizer(self, lr):
        trainable_param = self.model.parameters()
        optimizer = torch.optim.AdamW(trainable_param, lr=lr)
        self.optim_state = deepcopy(optimizer.state_dict())
        return optimizer 
    
    def be_episodic(self, *args, **kwargs):
        pass
    
    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.inference(*args, **kwargs)
    
    def step(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def inference(self, inputs):
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)
        return self.model(inputs)