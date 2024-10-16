import torch
from copy import deepcopy
from .base import confidence_filter, BaseTTAModule
from utils.tools import print, greedy_break
from clip.custom_clip import get_clip

class ZeroRLCF(BaseTTAModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.set_hyperparams(**kwargs)
        self.prepare_model(*args, **kwargs)
        self._freeze_params()

        # log hyperparameters
        self.name = "ZERO-RLCF"
        print("=> Hyperparameters: {}".format(self.get_hyperparams()))
        
    def set_hyperparams(self, **kwargs):    
        self.num_views = 64
        self.gamma = 0.3

    def get_hyperparams(self):
        return {
            "num_views": self.num_views,
            "gamma": self.gamma
        }

    # define an abstract interface here
    def prepare_model(self, *args, **kwargs):
        if kwargs.get("model") == "clip":
            self.model = get_clip(
                clip_arch=kwargs.get("arch"),
                pretrained=kwargs.get("pretrained"),
                device=kwargs.get("gpu"),
                ctx_init=kwargs.get("ctx_init"),
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

        # initialize the reward model
        self.reward_model = get_clip(
            clip_arch=kwargs.get("reward_arch"),
            pretrained=kwargs.get("reward_pretrained"),
            device=kwargs.get("gpu"),
            ctx_init=kwargs.get("ctx_init"),
            freeze_text=True,
            freeze_vision=True,
            cache_text_features=True,
            use_text_templates=bool(kwargs.get("use_templates")),
            maple_weights=kwargs.get("maple_weights"),
            maple_seed=kwargs.get("seed")
        )

    def _freeze_params(self):
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        print('=> Freezing all parameters.')

        for name, param in self.reward_model.named_parameters():
            param.requires_grad_(False)
        return

    def reset(self):
        self.model.reset()
        return
    
    def be_episodic(self, optimizer):
        pass

    def prepare_optimizer(self, lr):
        optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, self.model.parameters()), lr=lr)
        self.optim_state = deepcopy(optimizer.state_dict())
        return optimizer

    @torch.no_grad()
    def zero(self, views):
        
        # forward pass with the small model
        if hasattr(self.model, "text_features"):
            z_txt = self.model.text_features
        else:
            z_txt = self.model.get_text_features()
            self.model.text_features = z_txt

        z_img = self.model.get_image_features(views)

        # compute probabilities and confidence filter 
        l = z_img @ z_txt.t() # unscaled logits
        p = (l / self.model.temp).softmax(1) # probabilities
        _, idx = confidence_filter(l, p, top=self.gamma, return_idx=True) # retain most confident views

        # zero with the reward model
        if hasattr(self.reward_model, "text_features"):
            z_txt = self.reward_model.text_features
        else:
            z_txt = self.reward_model.get_text_features()
            self.reward_model.text_features = z_txt

        z_img = self.reward_model.get_image_features(views[idx])
        l_filt = z_img @ z_txt.t()

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
        

    def forward(self, *args, **kwargs):
        return self.zero(*args, **kwargs)    
    
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
