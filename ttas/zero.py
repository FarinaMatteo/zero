import torch

from copy import deepcopy
from .base import BaseTTAModule, confidence_filter
from clip.custom_clip import get_clip

from utils.tools import print, greedy_break




class Zero(BaseTTAModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prepare_model(*args, **kwargs)
        self._freeze_params()
        self.name = "ZERO"
        self.num_views = 64
        self.gamma = 0.1


    def prepare_model(self, *args, **kwargs):
        if kwargs.get("model") == "clip":
            self.model = get_clip(
                kwargs.get("arch"),
                kwargs.get("gpu"),
                kwargs.get("ctx_init"),
                cache_text_features=True,
                use_text_templates=bool(kwargs.get("use_templates")),
                freeze_text=True,
                freeze_vision=True,
                maple_weights=kwargs.get("maple_weights"),
                maple_seed=kwargs.get("seed")
            )
        else:
            raise NotImplementedError(f"{kwargs.get('model')} not implemented.")
        print("=> Model created: visual backbone {}".format(kwargs.get("arch")))


    def _freeze_params(self):
        for name, param in self.model.named_parameters():
            if "proj" not in name:
                param.requires_grad_(False)
        print('=> Freezing all parameters.')
        return

    
    @torch.no_grad()
    def zero(self, views):
        # here, we assume that the 'inputs' are a batch of augmented images,
        # where the first image is the original image and the rest are augmented\

        # compute text features (only once)
        if hasattr(self, "z_txt"):
            z_txt = self.z_txt
        else:
            z_txt = self.model.get_text_features()
            self.z_txt = z_txt
        
        # forward views online
        z_img = self.model.get_image_features(views) 

        # compute probabilities and confidence filter 
        l = z_img @ z_txt.t() # unscaled logits
        p = (l / self.model.temp).softmax(1) # probabilities
        l_filt = confidence_filter(l, p, top=self.gamma) # retain most confident views
        
        # zero-out the temperature, marginalize and predict
        zero_temp = torch.finfo(l_filt.dtype).eps
        p_bar = (l_filt / zero_temp).softmax(1).sum(0) # marginalize
        
        # check if we have to break ties in some way
        max_counts, scalar_pred = torch.max(p_bar, dim=-1)
        ties = [scalar_pred]
        for i in range(len(p_bar)):
            if i == scalar_pred: continue
            if p_bar[i] == max_counts: ties.append(i)

        # if so, break ties greedily
        if len(ties) > 1:
            k = int(views.size(0) * self.gamma)
            scalar_pred = greedy_break(ties, l[k:], device=l.device)
            p_bar[scalar_pred]+=1
        
        # need to unsqueeze for compatibility with the 'accuracy' function
        return p_bar.unsqueeze(0)
    

    def forward(self, images):
        return self.zero(images)
        
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

